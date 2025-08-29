import os
from typing import Collection, List, Union

import numpy as np

from .unet_segmentator import UNetSegmentation
from ..networks.unets import UNetv1


class HybridSegmentation(UNetSegmentation):
    """
    A segmentator that combines Unet with watershed
    """

    supported_setups = ["Family_Machine"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the UNetSegmentation using the base class init
        :*args: Arguments used for the base class init
        :**kwargs: Keyword arguments used for the basecalss init
        """

        # base class init
        super().__init__(*args, **kwargs)

    def _set_segmentation_method(self):
        """
        Sets the segmentation method according to the model_weights of the class
        """

        if self.model_weights == "watershed":
            self.segmentation_method = self.seg_method_watershed
        else:
            self.segmentation_method = self.seg_method_hybrid

    def _segs_for_selection(
        self, model_weights: List[Union[str, bytes, os.PathLike]], img: np.ndarray
    ):
        """
        Given the model weights, returns a selection of segmentation to use for the GUI selector
        :param model_weights: A list of model weights
        :param img: The image to segment
        :return: A list of segmentations starting with the watershed segmentation, i.e. 1 longer than model_weights
        """

        img_pad = self.pad_image(img)
        watershed_seg = self.segment_region_based(img, 0.16, 0.19)
        watershed_seg_pad = self.segment_region_based(img_pad, 0.16, 0.19)
        segs = [watershed_seg]
        for m in model_weights:
            model_pred = UNetv1(input_size=img_pad.shape[1:3] + (2,), inference=True)
            model_pred.load_weights(m)
            y_pred = model_pred.predict(
                np.concatenate([img_pad, watershed_seg_pad], axis=-1)
            )
            seg = (self.undo_padding(y_pred) > 0.5).astype(int)
            segs.append(seg)

        return segs

    def seg_method_hybrid(self, imgs_in: Collection[np.ndarray]):
        """
        Performs image segmentation with hybrid networkd and the selected model weights
        :param imgs_in: List of input images
        :return: List of segmentations
        """

        # pad the images
        imgs_pad = []
        for img in imgs_in:
            img_pad = self.pad_image(img)
            imgs_pad.append(img_pad)
        self.logger.info("Preparing watershed...")
        imgs_seg = self.seg_method_watershed(imgs_pad, min_val=0.15, max_val=0.17)
        imgs_seg = np.concatenate(imgs_seg, axis=0)

        # scale padded imgs
        imgs_pad = np.concatenate([self.scale_pixel_vals(img) for img in imgs_pad])

        # segments
        model_pred = UNetv1(input_size=imgs_pad.shape[1:3] + (2,), inference=True)
        model_pred.load_weights(self.model_weights)
        y_preds = model_pred.predict(
            np.concatenate([imgs_pad, imgs_seg], axis=-1), batch_size=1, verbose=1
        )

        # remove tha padding and transform to segmentation
        segs = []
        for i, y in enumerate(y_preds):
            seg = (self.undo_padding(y[None, ...]) > 0.5).astype(int)
            segs.append(seg)

        return segs
