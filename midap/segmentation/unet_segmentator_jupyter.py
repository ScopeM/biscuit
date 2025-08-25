import os
from pathlib import Path
from typing import Collection, Union, List

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries
from skimage import measure
from tqdm import tqdm

from .unet_segmentator import UNetSegmentation
from ..networks.unets import UNetv1
from ..utils import GUI_selector


class UNetSegmentationJupyter(UNetSegmentation):
    """
    A class that performs the image segmentation of the cells using a UNet
    """

    supported_setups = ["Jupyter"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the UNetSegmentation using the base class init
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

            # Get all the labels
            labels = ['watershed']
            model_weights = list(Path(self.path_model_weights).glob("*.h5"))
            labels += [mw.stem.replace("model_weights_", "") for mw in model_weights]

            self.all_segs_label = {}
            self.all_overl = {}
            for model_name in labels:
                self.model_weights = model_name
                self.segment_images_jupyter(imgs, model_name)
                
                # now we create an overlay of the image and the segmentation
                overl = [mark_boundaries(i, s, color=(1, 0, 0)) for i,s in zip(imgs, self.mask)]
                
                self.all_overl[model_name] = overl
                self.all_segs_label[model_name] = self.seg_label

    def set_segmentation_method_jupyter(self, path_to_cutouts: Union[str, bytes, os.PathLike]):
        """
        Performs the weight selection for the segmentation network. A custom method should use this function to set
        self.segmentation_method to a function that takes an input images and returns a segmentation of the image,
        i.e. an array in the same shape but with values only 0 (no cell) and 1 (cell)
        :param path_to_cutouts: The directory in which all the cutout images are
        """

        # check if we even need to select
        if self.model_weights is None:

            # get the image that is roughly in the middle of the stack
            list_files = np.sort(os.listdir(path_to_cutouts))
            # take the middle image (but round up, if there are only 2 we want the second)
            if len(list_files) == 1:
                ix_half = 0
            else:
                ix_half = int(np.ceil(len(list_files) / 2))

            path_img = list_files[ix_half]

            # scale the image and pad
            img = self.scale_pixel_vals(io.imread(os.path.join(path_to_cutouts, path_img)))
            print(img.max(), img.min())

            # Get all the labels
            labels = ['watershed']
            model_weights = list(Path(self.path_model_weights).glob("*.h5"))
            labels += [mw.stem.replace("model_weights_", "") for mw in model_weights]

            # create the segmentations
            segs = self._segs_for_selection(model_weights, img)

            # now we create a figures for the GUI
            self.segs = {}
            for seg, model_name in zip(segs, labels):
                
                # now we create an overlay of the image and the segmentation
                overl = mark_boundaries(img, seg, color=(1, 0, 0))
                self.segs[model_name] = overl

    
    def segment_images_jupyter(self, imgs, model_weights):
        """
        Sets the segmentation method according to the model_weights of the class
        """

        if self.model_weights == 'watershed':
            self.mask = self.seg_method_watershed(imgs)
        else:
            self.model_weights = os.path.join(self.path_model_weights, 'model_weights_' + model_weights + '.h5')
            self.mask = self.seg_method_unet(imgs)

        self.seg_label = np.array([measure.label(m) for m in self.mask])
        self.seg_bin = self.mask

