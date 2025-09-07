from __future__ import annotations
import numpy as np
import torch

from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

from . import base_segmentator

from bioimageio.core import load_model_description
from bioimageio.core.prediction import create_prediction_pipeline, predict


class BMZSegmentationJupyter(base_segmentator.SegmentationPredictor):
    """
    Hard-wired BioImage Model Zoo models with fixed input handling:
      - serious-lobster : axes order b c y x; y=x=256, b=1, c=1
      - affable-shark   : axes order b c y x; y,x >= 64 and multiples of 16, b=1, c=1

    Returns:
      - self.seg_label 
      - self.seg_bin   
    """
    
    supported_setups = {"Jupyter"}

    DEFAULT_MODELS = ["bmz_serious_lobster", "bmz_affable_shark", "bmz_happy_elephant"]

  
    MODEL_REF = {
        "bmz_serious_lobster": "serious-lobster",
        "bmz_affable_shark": "affable-shark"
    }


    def __init__(self, path_model_weights=None, postprocessing=False,
                 model_weights=None, img_threshold=255, device=None):
        super().__init__(path_model_weights, postprocessing, model_weights, img_threshold)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache = {}     
        self.seg_label = None     
        self.seg_bin = None       
        self._segmentation_method = None  

    def set_segmentation_method(self, method=None, **kwargs):
        """
        Required by the abstract base class. Not used by this notebook.
        """
        self._segmentation_method = method


    
    #def _pipeline(self, model_name: str):
    #    if model_name not in self._pipe_cache:
    #        ref = self.MODEL_REF[model_name]
    #        res = load_model_description(ref)
    #        self._pipe_cache[model_name] = create_prediction_pipeline(res, devices=[self.device])
    #    return self._pipe_cache[model_name]


    def _get_rd_pp(self, model_name):
        if model_name not in self._cache:
            rd = load_model_description(self.MODEL_REF[model_name])
            pp = create_prediction_pipeline(rd, devices=[self.device])
            self._cache[model_name] = (rd, pp)
        return self._cache[model_name]


    # ----------------------- input preparation (hard-wired) -----------------
    def _prep_serious_lobster(self, img2d: np.ndarray):
        """
        serious-lobster requires: (b=1,c=1,y=256,x=256).
        - If smaller: pad bottom/right to 256.
        - If larger : center-crop to 256, remember offsets, embed result back later.
        """
        if img2d.ndim == 3 and img2d.shape[-1] == 1:
            img2d = img2d[..., 0]
        if img2d.ndim != 2:
            raise ValueError(f"Expected 2D (H,W) or (H,W,1), got {img2d.shape}")

        H0, W0 = map(int, img2d.shape)
        Ht, Wt = 256, 256
        info = {"mode": None, "H0": H0, "W0": W0}

        if H0 > Ht or W0 > Wt:  
            y0 = max(0, (H0 - Ht) // 2)
            x0 = max(0, (W0 - Wt) // 2)
            img_proc = img2d[y0:y0+Ht, x0:x0+Wt]
            info.update({"mode": "crop", "y0": y0, "x0": x0})
        else:                   
            py, px = Ht - H0, Wt - W0
            img_proc = np.pad(img2d, ((0, py), (0, px)), mode="constant")
            info.update({"mode": "pad", "py": py, "px": px})

        x_bcyx = img_proc.astype("float32")[None, None, ...]  # (1,1,256,256)
        return x_bcyx, "bcyx", info

    def _prep_affable_shark(self, img2d: np.ndarray):
        """
        affable-shark requires (b=1,c=1,y,x) with y,x >= 64 and multiples of 16.
        Pad bottom/right up to the nearest multiple of 16 (and at least 64).
        """
        if img2d.ndim == 3 and img2d.shape[-1] == 1:
            img2d = img2d[..., 0]
        if img2d.ndim != 2:
            raise ValueError(f"Expected 2D (H,W) or (H,W,1), got {img2d.shape}")

        H0, W0 = map(int, img2d.shape)

        def round_up(n, step, minimum):
            n2 = max(n, minimum)
            r = n2 % step
            return n2 if r == 0 else (n2 + (step - r))

        Ht = round_up(H0, step=16, minimum=64)
        Wt = round_up(W0, step=16, minimum=64)

        py, px = Ht - H0, Wt - W0
        img_proc = np.pad(img2d, ((0, py), (0, px)), mode="constant")

        x_bcyx = img_proc.astype("float32")[None, None, ...]  # (1,1,Ht,Wt)
        info = {"mode": "pad", "py": py, "px": px, "H0": H0, "W0": W0}
        return x_bcyx, "bcyx", info


    

   # ----------------------- output post-processing -------------------------
    def _embed_back(self, lab: np.ndarray, info: dict):
        """Undo pad/crop to get back to original (H0,W0)."""
        H0, W0 = info["H0"], info["W0"]
        if info["mode"] == "pad":
            py, px = info.get("py", 0), info.get("px", 0)
            if py: lab = lab[:-py, :]
            if px: lab = lab[:, :-px]
            return lab
        elif info["mode"] == "crop":
            y0, x0 = info["y0"], info["x0"]
            out = np.zeros((H0, W0), dtype=lab.dtype)
            Ht, Wt = lab.shape
            out[y0:y0+Ht, x0:x0+Wt] = lab
            return out
        return lab  


    def _extract_2d(self, arr, *, prefer_foreground: bool = False) -> np.ndarray:

        a = np.asarray(arr)
        # Drop all singleton dims (safe; no axis specified)
        a = np.squeeze(a)

        if a.ndim == 2:
            return a

        if a.ndim == 3:
            C = a.shape[0]
            if prefer_foreground:
                if C in (2, 3):
                    return a[1]  
                elif C > 1:
                    return a.mean(axis=0)  
                else:
                    return a[0]
            else:
                return a[0]

        while a.ndim > 2:
            a = a[0]
        return a


    def _extract_fg_boundary(self, arr):
        """
        Try to interpret `arr` as a (C,H,W) or (H,W,C) tensor with C in {2,3},
        returning (foreground_prob, boundary_prob) as float32 in [0,1].
        Heuristic: the boundary channel tends to have smaller mean (thin edges).
        """
        a = np.asarray(arr)
        a = np.squeeze(a)

        if a.ndim == 3 and a.shape[0] in (2, 3):
            ch = a
        elif a.ndim == 3 and a.shape[-1] in (2, 3):
            ch = np.moveaxis(a, -1, 0)
        else:
            return None, None

        means = [float(ci.mean()) for ci in ch]
        b_ix = int(np.argmin(means))
        f_ix = int(np.argmax(means))

        fg = ch[f_ix].astype("float32")
        bd = ch[b_ix].astype("float32")

    # normalize to [0,1] (robustly)
        def _norm(x):
            x = x - np.nanmin(x)
            rng = np.nanmax(x) - np.nanmin(x)
            return (x / (rng + 1e-8)).astype("float32")
        return _norm(fg), _norm(bd)


    def _instances_from_fg_boundary(self, fg, bd):
        """
        Build instance labels from foreground & boundary probability maps using watershed.
        - fg: foreground probability in [0,1]
        - bd: boundary probability in [0,1] (higher = stronger boundary)
        Returns uint32 label image with background 0.
        """
        fg = np.clip(fg, 0, 1).astype("float32")
        bd = np.clip(bd, 0, 1).astype("float32")

        try:
            thr = float(threshold_otsu(fg))
        except Exception:
            thr = 0.5
        mask = fg > thr
        if mask.sum() == 0:
            return np.zeros_like(fg, dtype=np.uint32)

        hi = min(0.9, max(thr + 0.15 * (1 - thr), thr + 0.1))
        markers = label(fg > hi)
        if markers.max() == 0:
            markers = label(remove_small_objects(mask, 16))

        dist = ndi.distance_transform_edt(mask).astype("float32")
        elev = -(dist - 2.5 * bd)   # 2.5 weights boundary strength; tweak if needed

        lab = watershed(elev, markers=markers, mask=mask)
        lab = remove_small_objects(lab, 25)  # drop tiny fragments
        return lab.astype(np.uint32)

    
    
    
    def _to_labels(self, out_arrays: dict) -> np.ndarray:
        """
        Convert typical BioImage Model Zoo outputs to an instance label image.
         Prefers a 'masks' tensor if present in 'rdf outputs':
          - if it's integer-like -> cast to labels
          - else treat as probability map -> threshold -> connected components
        """
       
        def _as_2d_any(arr):
            return self._extract_2d(arr, prefer_foreground=False)

        def _as_2d_prob(arr):
            return self._extract_2d(arr, prefer_foreground=True).astype("float32")

        if "masks" in out_arrays:
            m = np.asarray(_as_2d_any(out_arrays["masks"]))
            if m.ndim == 2:
                if m.dtype.kind in "ui":
                    return m.astype(np.uint32)
                if m.dtype.kind == "f":
                    maxv = float(m.max()) if m.size else 0.0
                    if maxv > 1.5 and np.allclose(m, np.round(m), atol=1e-3):
                        return np.round(m).astype(np.uint32)
                    thr = float(m.mean() + 0.5 * m.std())
                    mask = remove_small_objects(m > thr, 16)
                    return label(mask).astype(np.uint32)
                    
        for key in ("labels", "instances", "instance_labels"):
            if key in out_arrays:
                arr2d = _as_2d_any(out_arrays[key])
                return np.assarray(arr2d.astype(np.uint32))

        for v in out_arrays.values():
            fg, bd = self._extract_fg_boundary(v)
            if fg is not None:
                return self._instances_from_fg_boundary(fg, bd)

        for key in ("prob", "probabilities", "foreground", "semantic"):
            if key in out_arrays:
                pm = _as_2d_prob(out_arrays[key])
                thr = float(pm.mean() + 0.5 * pm.std())
                mask = remove_small_objects(pm > thr, 16)
                return label(mask).astype(np.uint32)

        for v in out_arrays.values():
            if isinstance(v, np.ndarray):
                a2d = _as_2d_prob(v)
                thr = float(a2d.mean() + 0.5 * a2d.std())
                mask = remove_small_objects(a2d > thr, 16)
                return label(mask).astype(np.uint32)

        raise RuntimeError("BMZ model produced no array outputs to convert.")


    def _input_key_from_rd(self, rd):
        """Return the input tensor key ('id' or 'name'). Useful when determining labels"""
        inp = rd.inputs[0]
        return getattr(inp, "id", None) or getattr(inp, "name", None) or "input0"


    def run_image_stack_jupyter(self, imgs, model_name, clean_border=False):
        rd, pp = self._get_rd_pp(model_name)
        bmz_id = self.MODEL_REF[model_name]
        #input_id = rd.inputs[0].id  
        input_id = self._input_key_from_rd(rd)

        seg_lab, seg_bin = [], []
        for im in imgs:
            im2d = np.asarray(im)
            if bmz_id == "serious-lobster":
                x_bcyx, axes, info = self._prep_serious_lobster(im2d)
            elif bmz_id == "affable-shark":
                x_bcyx, axes, info = self._prep_affable_shark(im2d)
            else:
                raise RuntimeError(f"Unknown hard-wired BMZ id: {bmz_id}")

            sample = predict(model=pp, inputs={input_id: x_bcyx})
            arrays = sample.as_arrays()  

            lab = self._to_labels(arrays)
            lab = self._embed_back(lab, info)

            seg_lab.append(lab.astype(np.uint32))
            seg_bin.append((lab != 0).astype(np.uint8))

        self.seg_label = seg_lab
        self.seg_bin   = seg_bin


    def cleanup(self):
        # Close and drop cached BMZ pipelines (free GPU memory)
        try:
            for k, (rd, pp) in list(self._cache.items()):
                try:
                    # many runtimes don't have .close(), so just try
                    if hasattr(pp, "close"):
                        pp.close()
                except Exception:
                    pass
            self._cache.clear()
        except Exception:
            pass

        # Torch GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


    #def cleanup(self):
    #    if torch.cuda.is_available():
    #        torch.cuda.empty_cache()
