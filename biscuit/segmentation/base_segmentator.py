import os
import re
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import skimage.io as io
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from tqdm import tqdm

# Added for GPU configuration
try:
    import tensorflow as tf
except ImportError:
    # TensorFlow is not strictly required for all segmentators (e.g. Cellpose/Omnipose)
    # so we import it conditionally and fall back to CPU if it is missing.
    tf = None

from ..utils import get_logger

# get the logger we readout the variable or set it to max output
if "__VERBOSE" in os.environ:
    loglevel = int(os.environ["__VERBOSE"])
else:
    loglevel = 7
logger = get_logger(__file__, loglevel)


class SegmentationPredictor(ABC):
    """
    A class that performs the image segmentation of the cells
    """

    # this logger will be shared by all instances and subclasses
    logger = logger

    def __init__(
        self,
        path_model_weights: Union[str, bytes, os.PathLike],
        postprocessing: bool,
        div=16,
        connectivity=1,
        model_weights: Union[str, bytes, os.PathLike, None] = None,
        img_threshold=1.0,
    ):
        """
        Initializes the SegmentationPredictor instance
        :param path_model_weights: Path to the model weights
        :param postprocessing: A flag for the postprocessing
        :param div: Divisor used for the padding of the images. Images will be padded to next higher number that is
                    divisible by div
        :param connectivity: The connectivity used for the segmentation, see skimage.measure.label
        :param model_weights: Weights of the models to use, can be used to set the segmentation method
        :param img_threshold: Threshold for the images, all values brighter than this will be capped, defaults to 1.0,
                              which means no thresholding
        """

        # set the params
        self.path_model_weights = path_model_weights
        self.postprocessing = postprocessing
        self.div = div
        self.connectivity = connectivity
        self.threshold = img_threshold

        # This variable is used in case custom methods do not want the images padded (default)
        self.require_padding = False

        # params that will be set later
        self.model_weights = model_weights
        self.segmentation_method = None

        # ------------------------------------------------------------------
        # GPU handling
        # ------------------------------------------------------------------
        # For segmentators that rely on TensorFlow (e.g. UNet or StarDist), we
        # explicitly check whether a compatible GPU is available and, if so,
        # make sure TensorFlow is allowed to allocate memory on it.  Even if
        # the user only runs Cellpose-based models, performing this check is
        # harmless and guarantees a consistent behaviour across all back-ends.

        self.gpu_available = False
        if tf is not None:
            gpus = tf.config.list_physical_devices("GPU")
            if len(gpus) > 0:
                self.gpu_available = True
                try:
                    # Enable dynamic memory growth – safer than allowing TF to
                    # pre-allocate all GPU memory, especially in shared setups.
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.info(
                        f"TensorFlow will run on GPU(s): {[g.name for g in gpus]}"
                    )
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been
                    # initialised. If that already happened, we log the issue
                    # and silently continue – TF will still use the GPU.
                    self.logger.warning(
                        f"Could not set TensorFlow GPU memory growth: {e}"
                    )
            else:
                self.logger.info("No compatible GPU detected – TensorFlow will run on CPU.")
        else:
            # TensorFlow not installed – relevant for Omni/Cellpose paths that
            # rely on PyTorch instead. GPU management for those is handled in
            # the respective segmentators.
            self.logger.debug("TensorFlow not available; skipping GPU setup.")

    def run_image_stack_jupyter(self, imgs, model_weights, clean_border: bool):
        """
        Performs image segmentation, postprocessing and storage for all images found in channel_path
        :param channel_path: Directory of the channel used for the analysis
        """

        # segement all images
        self.segment_images_jupyter(imgs, model_weights)


    def run_image_stack(self, channel_path: Union[str, bytes, os.PathLike], clean_border: bool):
        """
        Performs image segmentation, postprocessing and storage for all images found in channel_path
        :param channel_path: Directory of the channel used for the analysis
        """
        path_cut = os.path.join(channel_path, "cut_im")
        path_seg = os.path.join(channel_path, "seg_im")
        path_seg_bin = os.path.join(channel_path, "seg_im_bin")

        # get all the images to segment
        path_imgs = np.sort(os.listdir(path_cut))

        # set the segmentation method if necessary
        if self.segmentation_method is None:
            self.set_segmentation_method(path_cut)

        # We read in all the images
        self.logger.info("Reading in images...")
        imgs = []
        for p in tqdm(path_imgs):
            imgs.append(io.imread(os.path.join(path_cut, p)))

        # segement all images
        self.logger.info("Segmenting images...")
        segs = self.segmentation_method(imgs)

        self.logger.info("Postprocessing and storage...")
        self.num_cells = []
        for seg, p in zip(segs, path_imgs):
            # postprocessing
            if self.postprocessing:
                seg = self.postprocess_seg(seg)

            # remove borders from the segmentation
            if clean_border:
                seg = clear_border(seg)

            # label in case no post processing or border removal
            seg = label(seg, connectivity=self.connectivity)

            self.num_cells.append(len(np.unique(seg)) - 1)

            # save individual image
            os.makedirs(path_seg, exist_ok=True)
            label_fname = re.sub("(_cut.tif|_cut.png|.tif)", "_seg.tif", p)
            io.imsave(
                os.path.join(path_seg, label_fname),
                seg.astype(np.uint16),
                check_contrast=False,
            )
            seg_fname = re.sub("(_cut.tif|_cut.png|.tif)", "_seg_bin.png", p)
            os.makedirs(path_seg_bin, exist_ok=True)
            io.imsave(
                os.path.join(path_seg_bin, seg_fname),
                255 * (seg > 0).astype(np.uint8),
                check_contrast=False,
            )

    def postprocess_seg(self, seg: np.ndarray):
        """
        Performs postprocessing on a segmentation, e.g. remove segmentations that are too small and area closing
        :param seg: The input segmentation
        :returns: the processed segmentation
        """

        # remove small and big particels which are not cells
        label_objects = label(seg, connectivity=self.connectivity)
        sizes = np.bincount(label_objects.ravel())
        reg = regionprops(label_objects)
        areas = [r.area for r in reg]

        # We take everything that is larger than 1% of the average size
        min_size = np.mean(areas) * 0.01
        mask_sizes = sizes > min_size
        mask_sizes[0] = 0
        # we multiply the labels to get a labelled image back
        img_filt = (mask_sizes[label_objects] > 0).astype(int) * label_objects

        return img_filt

    def scale_pixel_vals(self, img: np.ndarray):
        """
        Applies thresholding to and image (defined in init) and then scales the values of the pixels of an image such
        that they are between 0 and 1.
        :param img: The input image as array
        :returns: The images with pixels scales between 0 and 1
        """

        img = np.array(img)
        img = np.clip(img, img.min(), self.threshold * img.max())
        return (img - img.min()) / (img.max() - img.min())

    @abstractmethod
    def set_segmentation_method(self, path_to_cutouts):
        """
        This is an abstract method forcing subclasses to implement it
        """
        pass

    def cleanup(self):
        """
        Release GPU (and CPU) memory that may have been allocated by the
        deep-learning back-ends after a prediction run.  Call this once a
        predictor instance is no longer needed.  The helper tries to be
        framework-agnostic and silently ignores missing libraries.
        """
        import gc
        # --- TensorFlow --------------------------------------------------
        try:
            import tensorflow as _tf
            _tf.keras.backend.clear_session()
        except Exception:
            # covers ImportError and any runtime errors if TF is not in use
            pass

        # --- PyTorch / Cellpose / Omnipose -------------------------------
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # On some systems this additionally frees cached blocks
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

        # Force Python garbage collection (especially important in
        # long-running Jupyter sessions where references can linger).
        gc.collect()
