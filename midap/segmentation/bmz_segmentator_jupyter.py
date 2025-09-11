from __future__ import annotations
import numpy as np
import torch

from skimage.measure import label
from skimage.morphology import remove_small_objects

from . import base_segmentator

from bioimageio.core import load_model_description
from bioimageio.core.prediction import create_prediction_pipeline, predict


class BMZSegmentationJupyter(base_segmentator.SegmentationPredictor):
    """
    Hard-wired BioImage Model Zoo models with fixed input handling:
      - jolly-duck : axes order b c y x; y=x=256 , b=1 , c=1 
    

    Returns:
      - self.seg_label 
      - self.seg_bin   
    """
    
    supported_setups = {"Jupyter"}

    DEFAULT_MODELS = ["BioImage.IO_conscientious_seashell", "BioImage.IO_jolly_duck", 
                      "BioImage.IO_idealistic_water_buffalo"]

  
    MODEL_REF = {
        "BioImage.IO_conscientious_seashell": "conscientious-seashell",
        "BioImage.IO_jolly_duck": "jolly-duck",
        "BioImage.IO_idealistic_water_buffalo": "idealistic-water-buffalo"
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


    def _get_rd_pp(self, model_name):
        if model_name not in self._cache:
            rd = load_model_description(self.MODEL_REF[model_name])
            pp = create_prediction_pipeline(rd, devices=[self.device])
            self._cache[model_name] = (rd, pp)
        return self._cache[model_name]


    # ----------------------- input preparation (hard-wired) -----------------

    def _prep_conscientious_seashell(self, img2d: np.ndarray):
        """Expect (b=1,c=3,y,x). Tile grayscale to RGB; keep H,W."""
        if img2d.ndim == 3 and img2d.shape[-1] == 1:
            img2d = img2d[..., 0]
        if img2d.ndim != 2:
            raise ValueError(f"Expected 2D (H,W) or (H,W,1), got {img2d.shape}")
        
        H0, W0 = map(int, img2d.shape)
        rgb = np.stack([img2d, img2d, img2d], axis=0).astype("float32")  # (3,H,W)
        x_bcyx = rgb[None, ...]  # (1,3,H,W)
        info = {"mode": None, "H0": H0, "W0": W0}
        return x_bcyx, "bcyx", info


    def _prep_jolly_duck(self, img2d: np.ndarray):
        """Require (b=1,c=1,256,256); pad/crop to 256×256; float32."""
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
            img_proc = np.pad(img2d, ((0, py), (0, px)), mode="reflect")
            info.update({"mode": "pad", "py": py, "px": px})

        x_bcyx = img_proc.astype("float32")[None, None, ...]  # (1,1,256,256)
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
    

    def _to_labels(self, out_arrays: dict) -> np.ndarray:
        import numpy as np
        from skimage.filters import threshold_otsu
        from skimage.morphology import remove_small_objects, binary_closing
        from skimage.segmentation import watershed
        from skimage.measure import label as cc_label
        from scipy import ndimage as ndi

        a = np.asarray(out_arrays["output0"])
        if a.ndim == 4 and a.shape[0] == 1:  
            a = a[0]
        a = np.squeeze(a)  

        if a.ndim == 2:
            body = a.astype(np.float32)
            boundary = None
        elif a.ndim == 3:
            body = a[0].astype(np.float32)
            boundary = a[1].astype(np.float32) if a.shape[0] >= 2 else None
        else:
            raise RuntimeError(f"Unexpected output shape: {a.shape}")

        body = np.nan_to_num(body, nan=0.0, posinf=0.0, neginf=0.0)
        if boundary is not None:
            boundary = np.nan_to_num(boundary, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2 channel case (like idealistic-water-buffalo)
        if boundary is not None:
            try:
                t_body = float(threshold_otsu(body))
            except Exception:
                t_body = float(body.mean() + 0.5 * body.std())
            fg = body > t_body
            if not fg.any():
                return np.zeros_like(body, dtype=np.uint32)
            try:
                hi = float(np.percentile(body[fg], 35.0))
                t_seed = max(t_body, hi)
            except Exception:
                t_seed = float(max(t_body, body[fg].mean() + body[fg].std()))
            seeds = cc_label(body > t_seed)
            if seeds.max() == 0:            
                seeds = cc_label(fg) 
            lab = watershed(boundary, markers=seeds, mask=fg)
            lab = remove_small_objects(lab, 16)
            return lab.astype(np.uint32)

        # single-channel case (like jolly-duck)
        try:
            thr = float(threshold_otsu(body))
        except Exception:
            thr = float(body.mean() + 0.5 * body.std())
        mask = body > thr
        mask = ndi.binary_fill_holes(mask)
        mask = binary_closing(mask, footprint=np.ones((3, 3), dtype=bool))
        mask = remove_small_objects(mask, 16)
        return cc_label(mask).astype(np.uint32)
    
    


#    def _to_labels(self, out_arrays: dict) -> np.ndarray:

#        from skimage.filters import threshold_otsu
#        from skimage.morphology import binary_closing, square
#        from scipy.ndimage import binary_fill_holes
#        import numpy as np


#        if "output0" not in out_arrays:
#            raise RuntimeError("BMZ model produced no 'output0' to convert.")


#        def _as_2d_prob(arr):
#            a = self._extract_2d(arr, prefer_foreground=True).astype("float32")
#            if not np.isfinite(a).all():
#                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
#            return a

    
#        if "output0" in out_arrays:
#            #p = np.asarray(out_arrays["output0"])
#            #p2d = self._extract_2d(p, prefer_foreground=True).astype("float32")

#            p = _as_2d_prob(np.asarray(out_arrays["output0"]))

#            if p.size == 0 or float(p.max()) == float(p.min()):
#                return np.zeros_like(p, dtype=np.uint32)
            
#            try:
#                thr = float(threshold_otsu(p))
#            except Exception:
#                thr = float(p.mean() + 0.5 * p.std())

#            mask = p > thr
#            mask = binary_fill_holes(mask)
#            mask = binary_closing(mask, footprint=square(3))
#            mask = remove_small_objects(mask, min_size=16)
#            return label(mask).astype(np.uint32)
              


    def _input_key_from_rd(self, rd):
        """Return the input tensor key ('id' or 'name'). Useful when determining labels"""
        inp = rd.inputs[0]
        return getattr(inp, "id", None) or getattr(inp, "name", None) or "input0"


    def run_image_stack_jupyter(self, imgs, model_name, clean_border=False):
        rd, pp = self._get_rd_pp(model_name)
        bmz_id = self.MODEL_REF[model_name]
        input_id = self._input_key_from_rd(rd)

        seg_lab, seg_bin = [], []
        for im in imgs:
            im2d = np.asarray(im)
            if bmz_id == "conscientious-seashell":
                x_bcyx, _, info = self._prep_conscientious_seashell(im2d)
            elif bmz_id == "jolly-duck":
                x_bcyx, _, info = self._prep_jolly_duck(im2d)  
            elif bmz_id == "idealistic-water-buffalo":
                x_bcyx, _, info = self._prep_jolly_duck(im2d)  #NOTE: buddalo shares same prep as duck  
            else:
                raise RuntimeError(f"Unknown hard-wired BMZ id: {bmz_id}")

            sample = predict(model=pp, inputs={input_id: x_bcyx})
            arrays = np.asarray(sample.as_arrays()["output0"])
            
            #p = self._embed_back(arrays, info)             
            
            #arrays = dict(arrays); arrays["output0"] = p
            #lab = self._to_labels(arrays)

            arrays = np.squeeze(arrays)                            # -> (C,256,256) or (256,256)


            if arrays.ndim == 2:
                arrays = embed_back(arrays.astype("float32"), info)
            elif arrays.ndim == 3:
                arrays = np.stack([embed_back(arrays[c].astype("float32"), info)
                              for c in range(arrays.shape[0])], axis=0)
            else:
                raise RuntimeError(f"Unexpected output0 shape: {arrays.shape}")

            y = {"output0": arrays}
            lab = to_labels(y)
            
            #lab = self._embed_back(arrays, info)
            #lab = self._to_labels(lab)
            
            if bmz_id == "conscientious-seashell":
                H0, W0 = int(info["H0"]), int(info["W0"])
                h, w = lab.shape[:2]
                if (h, w) != (H0, W0):
                    if h >= H0 and w >= W0:
                        y0 = max(0, (h - H0) // 2)
                        x0 = max(0, (w - W0) // 2)
                        lab = lab[y0:y0 + H0, x0:x0 + W0]
                    else:
                        py = max(0, H0 - h); px = max(0, W0 - w)
                        lab = np.pad(lab, ((0, py), (0, px)), mode="constant")

            
            seg_lab.append(lab.astype(np.uint32))
            seg_bin.append((lab != 0).astype(np.uint8))

        self.seg_label = seg_lab
        self.seg_bin   = seg_bin


    def cleanup(self):
        # Close and drop cached BMZ pipelines (free GPU memory)
        try:
            for k, (rd, pp) in list(self._cache.items()):
                try:
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

