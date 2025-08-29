"""
CellposeSegmentationJupyter
---------------------------
Light-weight Jupyter wrapper around Cellpose models (non-Omnipose).  The code
is a trimmed copy of *omni_segmentator_jupyter.py* with the sole relevant
behavioural change: model.eval is called with omni = False.
"""

import os
from typing import Union, List

import numpy as np
import skimage.io as io
from skimage.segmentation import mark_boundaries

from .cellpose_segmentator import CellposeSegmentation     # re-use helper funcs / logger

class CellposeSegmentationJupyter(CellposeSegmentation):
    """
    Segment images in a Jupyter notebook with the classic Cellpose models.
    """

    supported_setups = ["Jupyter"]

    # built-in Cellpose checkpoints that ship with cellpose-omni
    AVAILABLE_MODELS = [
        'cpsam'
    ]
    DEFAULT_MODELS = ['cpsam']   # first candidates in GUI

    # --------------------------------------------------------------------- #
    # 1.  Weight selection helper (interactive preview on a few cut-outs)   #
    # --------------------------------------------------------------------- #
    def set_segmentation_method_jupyter_all_imgs(
        self,
        path_to_cutouts: Union[str, os.PathLike, bytes],
    ):
        """
        Inspect a handful of cut-outs and let the user choose the model that
        looks best.  Behaviour identical to the Omnipose counterpart except
        that omni=False is passed to *model.eval*.
        """
        if self.model_weights is not None:      # nothing to do
            return

        # load the whole stack of cut-outs
        list_files = sorted(
            f for f in os.listdir(path_to_cutouts)
            if not f.startswith((".", "_"))
        )
        imgs = [self.scale_pixel_vals(io.imread(os.path.join(path_to_cutouts, f)))
                for f in list_files]

        # try every default model and collect quick overlays
        label_dict = {m: m for m in self.DEFAULT_MODELS
                      if m in self.AVAILABLE_MODELS}

        self.all_segs_label, self.all_overl = {}, {}
        for mdl_name, mdl_path in label_dict.items():
            model = self._build_cellpose_model(mdl_path, gpu=True)
            try:
                eval_imgs = ([np.stack([im, im], -1) for im in imgs]
                             if model.nchan == 2 and imgs[0].ndim == 2 else imgs)
                mask, _, _ = model.eval(
                    eval_imgs,
                    channels=[0, 0],
                    mask_threshold=-1,
                    flow_threshold=0,
                    transparency=True,
                    omni=False,          # ← DIFFERENCE
                    resample=True,
                    niter=20,
                    verbose=0,
                )
                seg = (np.asarray(mask) > 0).astype(int)
                overl = [mark_boundaries(i, s, color=(1, 0, 0))
                         for i, s in zip(imgs, seg)]

                self.all_segs_label[mdl_name] = mask
                self.all_overl[mdl_name]      = overl
            except ValueError:
                # Cellpose sometimes refuses extremely small masks – just skip
                continue

    # --------------------------------------------------------------------- #
    # 2.  Programmatic call used by SegmentationJupyter.run_image_stack     #
    # --------------------------------------------------------------------- #
    def segment_images_jupyter(self, imgs: List[np.ndarray], model_weights: str):
        """
        Predict a full stack in one go.  Apart from omni=False the call mirrors
        *OmniSegmentationJupyter.segment_images_jupyter*.
        """
        model = self._build_cellpose_model(model_weights, gpu=True)
        imgs  = [self.scale_pixel_vals(im) for im in imgs]

        try:
            #eval_imgs = ([np.stack([im, im], -1) for im in imgs]
            #             if model.nchan == 2 and imgs[0].ndim == 2 else imgs)
            mask, _, _ = model.eval(
                imgs,
                channels=[0, 0],
   
            )
        except ValueError:
            self.logger.warning("Cellpose failed, returning empty mask.")
            mask = np.zeros((len(imgs),) + imgs[0].shape, dtype=int)

        # ----------------------------------------------------------
        # Cellpose may return *list[ndarray]*.  Convert it into one
        # NumPy stack so that the dimensionality checks below work.
        # ----------------------------------------------------------
        mask = np.asarray(mask)

        # keep only first channel if mask is (H,W,2)
        if mask.ndim == 3 and mask.shape[-1] == 2:
            mask = mask[..., 0]

        self.seg_bin   = (mask > 0).astype(int)
        self.seg_label = mask