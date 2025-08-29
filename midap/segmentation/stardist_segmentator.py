import os
from pathlib import Path
from typing import Collection, Union, List

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from stardist.models import StarDist2D
from csbdeep.utils import normalize

from .base_segmentator import SegmentationPredictor
from ..utils import GUI_selector


class StarDistSegmentation(SegmentationPredictor):
    """
    A class that performs the image segmentation of the cells using a UNet
    """

    supported_setups = ["Family_Machine", "Mother_Machine"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the UNetSegmentation using the base class init
        :*args: Arguments used for the base class init
        :**kwargs: Keyword arguments used for the basecalss init
        """

        # base class init
        super().__init__(*args, **kwargs)

        self.labels = ["2D_versatile_fluo", "2D_paper_dsb2018"]

    def _segs_for_selection(
        self, model_weights: List[Union[str, bytes, os.PathLike]], img: np.ndarray
    ):
        """
        Given the model weights, returns a selection of segmentation to use for the GUI selector
        :param model_weights: A list of folders containing pretrained StarDist models
        :param img: The image to segment
        :return: A list of segmentations and corresponding labels
        """

        segs_labels = []
        for l in self.labels:
            model = StarDist2D.from_pretrained(l)
            mask, _ = model.predict_instances(normalize(img))
            seg = (mask > 0.5).astype(int)
            segs_labels.append(seg)

        segs_weights = []
        for m in model_weights:
            model = StarDist2D(None, name=str(m))
            mask, _ = model.predict_instances(normalize(img))
            seg = (mask > 0.5).astype(int)
            segs_weights.append(seg)

        segs_all = segs_labels + segs_weights
        labels_all = self.labels.copy()
        labels_all += [mw.stem.replace("model_weights_", "") for mw in model_weights]
        return segs_all, labels_all

    def set_segmentation_method(self, path_to_cutouts: Union[str, bytes, os.PathLike]):
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
            self.logger.info(f"The shape of the image is: {img.shape}")

            # get all trained models
            model_weights = [
                path
                for path in Path(self.path_model_weights).iterdir()
                if path.is_dir()
            ]
            # labels = ['2D_versatile_fluo', '2D_paper_dsb2018', '2D_versatile_he']

            # create the segmentations
            segs, model_names = self._segs_for_selection(model_weights, img)

            figures = []
            for s, l in zip(segs, model_names):
                # now we create a plot that can be used as a button image
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(img)
                ax.contour(s, [0.5], colors="r", linewidths=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(l)
                figures.append(fig)

            # Title for the GUI
            channel = os.path.basename(os.path.dirname(path_to_cutouts))
            # if we just got the chamber folder, we need to go one more up
            if channel.startswith("chamber"):
                channel = os.path.basename(
                    os.path.dirname(os.path.dirname(path_to_cutouts))
                )
            title = f"Segmentation Selection for channel: {channel}"

            # start the gui
            marked = GUI_selector(figures=figures, labels=model_names, title=title)

            self.model_weights = marked

        # helper functions for the seg method based on chosen weights
        def seg_method_name(imgs: Collection[np.ndarray]):
            """
            Performs the segmentation given a model name
            :param imgs: Images to segment
            :return: The segmented images
            """
            masks = []
            for img in imgs:
                img = self.scale_pixel_vals(img)
                model = StarDist2D.from_pretrained(self.model_weights)
                mask, _ = model.predict_instances(normalize(img))
                masks.append(mask)
            return np.stack(masks, axis=0)

        def seg_method_dir(imgs: Collection[np.ndarray]):
            """
            Performs the segmentation given a a path to a model directory
            :param imgs: Images to segment
            :return: The segmented images
            """
            masks = []
            for img in imgs:
                model = StarDist2D(None, name=str(self.model_weights))
                mask, _ = model.predict_instances(normalize(img))
                masks.append(mask)

            return np.stack(masks, axis=0)

        # set the segmentations method
        if self.model_weights in self.labels:
            self.segmentation_method = seg_method_name
        else:
            self.model_weights = os.path.join(
                self.path_model_weights, self.model_weights
            )
            self.segmentation_method = seg_method_dir
