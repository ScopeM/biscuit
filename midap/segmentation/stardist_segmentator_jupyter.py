import os
from pathlib import Path
from typing import Collection, Union, List

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage.segmentation import mark_boundaries
from skimage import measure
from stardist.models import StarDist2D
from csbdeep.utils import normalize

from .stardist_segmentator import StarDistSegmentation
from ..utils import GUI_selector


class StarDistSegmentationJupyter(StarDistSegmentation):
    """
    A class that performs the image segmentation of the cells using a UNet
    """

    supported_setups = ["Jupyter"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the StarDistSegmentationJupyter using the base class init
        :*args: Arguments used for the base class init
        :**kwargs: Keyword arguments used for the basecalss init
        """

        # base class init
        super().__init__(*args, **kwargs)

    def set_segmentation_method_jupyter_all_imgs(self, path_to_cutouts: Union[str, bytes, os.PathLike]):
        """
        Performs the weight selection for the segmentation network. A custom method should use this function to set
        self.segmentation_method to a function that takes an input images and returns a segmentation of the image,
        i.e. an array in the same shape but with values only 0 (no cell) and 1 (cell)
        :param path_to_cutouts: The directory in which all the cutout images are
        """

        # check if we even need to select
        if self.model_weights is None:

            # get the image that is roughly in the middle of the stack
            list_files = np.sort([f for f in os.listdir(path_to_cutouts) if not f.startswith((".", "_"))])

            # scale the image and pad
            imgs = []
            for f in list_files:
                img = self.scale_pixel_vals(io.imread(os.path.join(path_to_cutouts, f)))
                imgs.append(img)
            imgs = np.array(imgs)

            # Get all the labels
            labels = ['2D_versatile_fluo', '2D_paper_dsb2018']#, '2D_versatile_he']

            self.all_segs_label = {}
            self.all_overl = {}
            for model_name in labels:
                model = StarDist2D.from_pretrained(model_name)
                # predict, we only need the mask, see omnipose tutorial for the rest of the args
                mask = np.array([model.predict_instances(normalize(img))[0] for img in imgs])
                # omni removes axes that are just 1
                self.seg_bin = (mask > 0).astype(int)
                self.seg_label = mask
                
                # now we create an overlay of the image and the segmentation
                overl = [mark_boundaries(i, s, color=(1, 0, 0)) for i,s in zip(imgs, mask)]

                self.all_overl[model_name] = overl
                self.all_segs_label[model_name] = self.seg_label

    def segment_images_jupyter(self, imgs, model_name):
        """
        Sets the segmentation method according to the model_weights of the class
        """
        model = StarDist2D.from_pretrained(model_name)
                
        # predict, we only need the mask, see omnipose tutorial for the rest of the args
        mask = np.array([model.predict_instances(normalize(img))[0] for img in imgs])

        self.seg_bin = (mask > 0).astype(int)
        self.seg_label = mask


