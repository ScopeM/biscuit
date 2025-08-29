import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from cellpose_omni import models
from typing import Union

import torch

from .base_segmentator import SegmentationPredictor
from ..utils import GUI_selector
import platform


class OmniSegmentation(SegmentationPredictor):
    """
    A class that performs the image segmentation of the cells using a UNet
    """

    supported_setups = ["Family_Machine", "Mother_Machine"]

    # --------------------------------------------------------------
    # ❶  Library-provided and default model lists
    # --------------------------------------------------------------
    AVAILABLE_MODELS = [
        "bact_phase_omni", "bact_fluor_omni", "worm_omni",
        "worm_bact_omni", "worm_high_res_omni", "cyto2_omni",
        "plant_omni", "bact_phase_cp", "bact_fluor_cp", "plant_cp",
        "worm_cp", "cyto", "nuclei", "cyto2",
    ]
    DEFAULT_MODELS = ["nuclei","bact_fluor_omni","bact_phase_omni","worm_omni","worm_bact_omni"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the UNetSegmentation using the base class init
        :*args: Arguments used for the base class init
        :**kwargs: Keyword arguments used for the basecalss init
        """

        # base class init
        super().__init__(*args, **kwargs)

        if platform.processor() == "arm":
            self.gpu_available = torch.backends.mps.is_available()
        else:
            self.gpu_available = torch.cuda.is_available()

    def set_segmentation_method(self, path_to_cutouts):
        """
        Performs the weight selection for the segmentation network. A custom method should use this function to set
        self.segmentation_method to a function that takes an input images and returns a segmentation of the image,
        i.e. an array in the same shape but with values only 0 (no cell) and 1 (cell)
        :param path_to_cutouts: The directory in which all the cutout images are
        """

        if self.model_weights is None:
            self.logger.info("Selecting weights...")

            # get the image that is roughly in the middle of the stack
            list_files = np.sort(os.listdir(path_to_cutouts))
            # take the middle image (but round up, if there are only 2 we want the second)
            if len(list_files) == 1:
                ix_half = 0
            else:
                ix_half = int(np.ceil(len(list_files) / 2))

            path_img = list_files[ix_half]

            # scale the image and pad
            img = self.scale_pixel_vals(
                io.imread(os.path.join(path_to_cutouts, path_img))
            )

            # display different segmentation models
            # start with the class default models
            label_dict = {m: m for m in self.DEFAULT_MODELS
                          if m in self.AVAILABLE_MODELS}
            figures = []
            '''for model_name, model_path in label_dict.items():
                self.logger.info("Try model: " + str(model_name))
                model = self._build_cellpose_model(
                    model_path, gpu=self.gpu_available
                )

                # predict – adjust image if the model expects 2 channels
                test_img = (np.stack([img, img], -1)
                            if model.nchan == 2 and img.ndim == 2 else img)
                mask, _, _ = model.eval(
                    test_img,
                    channels=[0, 0],
                    rescale=None,
                    mask_threshold=-1,
                    transparency=True,
                    flow_threshold=0,
                    omni=True,
                    resample=False,
                    niter=20,
                    verbose=0,
                )
                # omni removes axes that are just 1
                seg = (mask > 0.5).astype(int)

                # now we create a plot that can be used as a button image
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(img)
                ax.contour(seg, [0.5], colors="r", linewidths=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(model_name)
                figures.append(fig)
                '''

            # Title for the GUI
            channel = os.path.basename(os.path.dirname(path_to_cutouts))
            # if we just got the chamber folder, we need to go one more up
            if channel.startswith("chamber"):
                channel = os.path.basename(
                    os.path.dirname(os.path.dirname(path_to_cutouts))
                )
            title = f"Segmentation Selection for channel: {channel}"

            # start the gui
            marked = GUI_selector(
                figures=figures, labels=list(label_dict.keys()), title=title
            )

            # set weights
            self.model_weights = label_dict[marked]

        # helper function for the seg method
        model = self._build_cellpose_model(
            self.model_weights, gpu=self.gpu_available
        )

        def seg_method(imgs):
            # scale all the images
            #imgs = [self.scale_pixel_vals(img) for img in imgs]
            #if model.nchan == 2 and imgs[0].ndim == 2:        # duplicate chan
            #    imgs = [np.stack([im, im], axis=-1) for im in imgs]

            # we catch here ValueErrors because omni can fail at masking when there are no cells

            try:
                mask, _, _ = model.eval(
                    imgs,
                    channels=[0, 0],
                    rescale=None,
                    mask_threshold=-1,
                    transparency=True,
                    flow_threshold=0,
                    omni=True,
                    cluster=True,
                    resample=True,
                    tile= False,
                    niter= None,
                    augment= False,
                    affinity_seg= True,
                    verbose=False,
                )
            except ValueError as e:
                # collect some context to understand why it failed
                msg  = f"Omnipose ValueError: {e}\n"
                msg += f"  ‣ model expects nchan = {model.nchan}\n"
                msg += f"  ‣ imgs is {type(imgs)}, len = {len(imgs)}\n"
                msg += f"  ‣ first image shape  = {np.shape(imgs[0])}\n"
                msg += f"  ‣ img array shape   = {np.shape(imgs)}\n"
                msg += f"  ‣ first image dtype  = {np.asarray(imgs[0]).dtype}\n"
                self.logger.error(msg)
                print(msg)
                # fall back to empty mask so the pipeline can continue
                mask = np.zeros((len(imgs),) + imgs[0].shape, dtype=int)

            # keep only one channel if mask is (H, W, 2)
            if mask.ndim == 3 and mask.shape[-1] == 2:
                mask = mask[..., 0]

            self.seg_bin   = (np.array(mask) > 0).astype(int)
            self.seg_label = mask

            # add the channel dimension and batch if it was 1
            return mask

        # set the segmentations method
        self.segmentation_method = seg_method

    # --------------------------------------------------------------
    # helper: build Cellpose / Omnipose model with correct nchan
    # --------------------------------------------------------------
    def _build_cellpose_model(
        self, model_id: Union[str, os.PathLike], *, gpu: bool
    ):
        """
        Create a CellposeModel with nchan adjusted to the checkpoint.
        Falls back to 1-channel if the weight file cannot be inspected.
        """
        built_in_two_chan = {
            "cyto", "cyto2", "cytotorch_0",
            "bact_phase_cp", "bact_fluor_cp",
        }

        model_id = str(model_id)
        #nchan = 1

        if Path(model_id).is_file():                      # custom file
            try:
                ckpt = torch.load(model_id, map_location="cpu")
                # take the *largest* in-channel dimension among all
                # 4-D tensors; Cellpose checkpoints contain only 1- or
                # 2-channel kernels, so this reliably gives 1 or 2.
                nchan = max(
                    tensor.shape[1]
                    for tensor in ckpt["state_dict"].values()
                    if tensor.ndim == 4
                )
            except (StopIteration, KeyError):
                # checkpoint was read but tensor not found – fallback
                stem = Path(model_id).stem
                if stem in built_in_two_chan:
                    nchan = 2
            except Exception:
                # file not readable => use filename heuristic
                stem = Path(model_id).stem
                if stem in built_in_two_chan:
                    nchan = 2
            return models.CellposeModel(
                gpu=gpu, nchan=nchan, pretrained_model=model_id
            )

        #if model_id in built_in_two_chan:                 # built-in name
        #    nchan = 2

        return models.CellposeModel(gpu=gpu, model_type=model_id)
