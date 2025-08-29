import os
from pathlib import Path
from typing import Collection, Union, List

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage.segmentation import mark_boundaries
from cellpose_omni import models

from .omni_segmentator import OmniSegmentation
from ..utils import GUI_selector


class OmniSegmentationJupyter(OmniSegmentation):
    """
    A class that performs the image segmentation of the cells using a UNet
    """

    supported_setups = ["Jupyter"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the OmniSegmentationJupyter using the base class init
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

            # start with the class-wide default models
            label_dict = {m: m for m in self.DEFAULT_MODELS
                          if m in self.AVAILABLE_MODELS}

            self.all_segs_label = {}
            self.all_overl = {}
            for model_name, model_path in label_dict.items():
                print(model_name, model_path)
                model = self._build_cellpose_model(model_path, gpu=True)

                # predict, we only need the mask, see omnipose tutorial for the rest of the args
                try:
                    eval_imgs = ( [np.stack([im, im], -1) for im in imgs]
                                  if model.nchan == 2 and imgs[0].ndim == 2
                                  else imgs )
                    mask, _, _ = model.eval(
                        eval_imgs,
                        channels=[0, 0],
                        rescale=None,
                        mask_threshold=-1,
                        transparency=True,
                        flow_threshold=0,
                        omni=True,
                        resample=True,
                        niter=20,
                        verbose=0,
                    )
                                    # omni removes axes that are just 1

                    self.seg_bin = (np.array(mask) > 0).astype(int)
                    self.seg_label = mask
                    
                    # now we create an overlay of the image and the segmentation
                    overl = [mark_boundaries(i, s, color=(1, 0, 0)) for i,s in zip(imgs, self.seg_bin)]
                    self.all_overl[model_name] = overl
                    self.all_segs_label[model_name] = self.seg_label

                except ValueError: #in case KNN is throwing an error
                    pass


    def segment_images_jupyter(self, imgs, model_weights):
        # helper function for the seg method
        model = self._build_cellpose_model(model_weights, gpu=True)

        # --------- diagnostics ------------------------------------
        # print("DEBUG – OmniSegmentationJupyter")
        # print("  len(imgs):      ", len(imgs))
        # print("  model.nchan:    ", model.nchan)
        # print("  imgs[0].shape:  ", imgs[0].shape)
        # print("  imgs[0].dtype:  ", np.asarray(imgs[0]).dtype)
        # print("  imgs array shape", np.shape(imgs))
        # print("--------------------------------------------------------")

        # scale all the images
        imgs = [self.scale_pixel_vals(img) for img in imgs]
        #if model.nchan == 2 and imgs[0].ndim == 2:
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
        except ValueError:
            self.logger.warning('Segmentation failed, returning empty mask!')
            mask = np.zeros((len(imgs), ) + imgs[0].shape, dtype=int)

        self.seg_bin = (np.array(mask) > 0).astype(int)
        self.seg_label = mask

        # add the channel dimension and batch if it was 1

        # set the segmentations method
        #self.segmentation_method = seg_method


