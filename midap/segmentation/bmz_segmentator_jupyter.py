from __future__ import annotations
import numpy as np
import torch
from skimage.measure import label
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi

from . import base_segmentator
from bioimageio.core import load_model_description
from bioimageio.core.prediction import create_prediction_pipeline

class BMZSegmentationJupyter(base_segmentator.SegmentationPredictor):
    """
    Jupyter backend for exactly two hard-wired BioImage Model Zoo models.
    """
    supported_setups = {"Jupyter"}

    DEFAULT_MODELS = ["bmz_merry_gorilla1", "bmz_merry_gorilla2"]

  
    MODEL_REF = {
        "bmz_merry_gorilla1": "merry-gorilla",
        "bmz_merry_gorilla2": "merry-gorilla",
    }

    AXES_HINT = "yx"

    def __init__(self, path_model_weights=None, postprocessing=False,
                 model_weights=None, img_threshold=255, device=None):
        super().__init__(path_model_weights, postprocessing, model_weights, img_threshold)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe_cache = {}     
        self.seg_label = None     
        self.seg_bin = None       


    def _to_labels(self, out):
        if isinstance(out, dict):
            for k in ("labels", "instances", "instance_labels"):
                if k in out and out[k] is not None:
                    return np.asarray(out[k]).astype(np.uint32)
            for k in ("prob", "probabilities", "foreground", "semantic"):
                if k in out and out[k] is not None:
                    pm = np.asarray(out[k])
                    if pm.ndim == 3 and pm.shape[0] in (2, 3):
                        pm = pm[1]  # choose foreground channel if present
                    mask = pm > (pm.mean() + 0.5 * pm.std())
                    mask = remove_small_objects(mask, 16)
                    return label(mask).astype(np.uint32)
            if "boundary" in out and "foreground" in out:
                fore = np.asarray(out["foreground"]) > 0.5
                dist = ndi.distance_transform_edt(fore)
                markers = label(dist > np.percentile(dist[fore], 60))
                return ndi.watershed(-dist, markers, mask=fore).astype(np.uint32)
            for v in out.values():
                if isinstance(v, np.ndarray):
                    return label(v > v.mean()).astype(np.uint32)

        arr = np.asarray(out)
        if arr.dtype.kind in "iu":
            return arr.astype(np.uint32)
        return label(arr > arr.mean()).astype(np.uint32)

    def _pipeline(self, model_name: str):
        if model_name not in self._pipe_cache:
            ref = self.MODEL_REF[model_name]
            res = load_model_description(ref)                 
            self._pipe_cache[model_name] = create_prediction_pipeline(res, devices=[self.device])
        return self._pipe_cache[model_name]

    def run_image_stack_jupyter(self, imgs, model_name, clean_border=False):

        pipe = self._pipeline(model_name)
        axes = self.AXES_HINT

        seg_lab, seg_bin = [], []
        for im in imgs:
            x = np.asarray(im)
            if x.ndim == 3 and x.shape[-1] == 1:    
                x = x[..., 0]
            out = pipe(x, axes=axes)                
            lab = self._to_labels(out)
            seg_lab.append(lab.astype(np.uint32))
            seg_bin.append((lab != 0).astype(np.uint8))

        self.seg_label = seg_lab
        self.seg_bin = seg_bin

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
