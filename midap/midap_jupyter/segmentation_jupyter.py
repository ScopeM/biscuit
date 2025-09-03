import os
import sys
import shutil
import time

from skimage import io
from pathlib import Path
from importlib import resources          # Python ≥ 3.9

#import matplotlib
#import ipympl
#if 'matplotlib.pyplot' not in sys.modules:
#    backend = os.environ.get('MATPLOTLIB_BACKEND', 'TkAgg')
#    import matplotlib
#    matplotlib.use(backend)
#else:
import matplotlib

#matplotlib.use(os.environ.get("MATPLOTLIB_BACKEND", "module://ipympl.backend_nbagg"))
import matplotlib.pyplot as plt

import mpl_interactions.ipyplot as iplt

import numpy as np
import pandas as pd
import glob

from midap.utils import get_inheritors
from midap.segmentation import *
from midap.segmentation import base_segmentator
from midap.apps import segment_cells

import ipywidgets as widgets
from ipywidgets import interactive, Text, Password, Button, Output
from matplotlib.widgets import RadioButtons
from PIL import Image
from ipyfilechooser import FileChooser
import IPython as ip
import subprocess
from IPython.display import display, clear_output

from typing import Union, List

from google.colab import data_table


class SegmentationJupyter(object):
    """
    A class that performs image processing and segmentation based on midap
    """

    def __init__(self, path: Union[str, os.PathLike]):
        """
        Initializes the SegmentationJupyter
        :path: path to folder containing images
        """
        self.path = path
        self.path_midap = str(Path(__file__).parent.resolve().parent.parent)

        # existing folders
        self.path_data_input = self.path + "/input_data/"
        self.path_data = self.path + "/raw_im/"

        # ensure a raw_im folder exists and, if absent, seed it with
        # example images that ship with MIDAP so the notebook has
        # something to work with on a fresh installation.
        if not os.path.isdir(self.path_data):
            os.makedirs(self.path_data, exist_ok=True)

            # copy bundled example TIFFs (if any) into the new directory.
            # The files are installed as midap/data_examples/*
            import importlib.resources as resources
            try:
                example_dir = resources.files("midap").joinpath("data_examples")
            except AttributeError:        # Py < 3.9 fallback
                import importlib_resources as resources
                example_dir = resources.files("midap").joinpath("data_examples")

            if example_dir.exists():
                for f in example_dir.iterdir():
                    if f.is_file():
                        shutil.copy(f, self.path_data)

        # folders created by class
        self.path_cut_base = Path(self.path).joinpath("cut_im/")
        self.path_seg_base = Path(self.path).joinpath("seg_im/")
        os.makedirs(self.path_cut_base, exist_ok=True)
        os.makedirs(self.path_seg_base, exist_ok=True)

    def get_input_dir(self):
        """
        Extracts input directory.
        """
        self.fc_file = FileChooser(self.path)
        self.fc_file.show_only_dirs = True
        self.fc_file.layout ={"width": "600px"}

    def get_input_files(self, path):
        """
        Extracts input file names.
        """
        self.file_selection = widgets.SelectMultiple(
            options=os.listdir(path),
            description="Files",
            disabled=False,
            layout={"height": "250px", "width": "600px"},
        )

        self.button = Button(description="Select")
        self.output = Output()

        def on_button_clicked(b):
            with self.output:
                self.chosen_files = self.file_selection.label
                self.chosen_dir = self.fc_file.selected
                #self.load_input_image()

        self.button.on_click(on_button_clicked)
        ip.display.display(self.file_selection)
        ip.display.display(self.button)

    def load_input_image(self, image_stack=False):
        """
        Loads selected images, equalises their spatial dimensions and
        extracts image dimensions for further processing.

        Workflow:
        1) Probe each file header to determine its height/width without
           actually loading the full pixel data (fast & memory-efficient).
        2) Load the images, zero-padding them so that every image attains
           the maximum height/width observed across the selection.
        3) Continue with the original dimension bookkeeping logic.
        """

        # ------------------------------------------------------------------
        # 1) Determine maximum height & width across all chosen images ------
        # ------------------------------------------------------------------
        sizes_hw = []  # list of tuples (H, W)
        for f in self.chosen_files:
            with Image.open(Path(self.chosen_dir).joinpath(f)) as im:
                sizes_hw.append(im.size[::-1])  # PIL.Image.size → (W, H) → (H, W)
        max_h = max(h for h, _ in sizes_hw)
        max_w = max(w for _, w in sizes_hw)

        # ------------------------------------------------------------------
        # 2) Load images and zero-pad them to (max_h, max_w) ----------------
        # ------------------------------------------------------------------
        self.imgs = []
        for f in self.chosen_files:
            path_chosen_img = Path(self.chosen_dir).joinpath(f)
            img = io.imread(path_chosen_img)
            h, w = img.shape[:2]
            pad_h = max_h - h
            pad_w = max_w - w
            pad_spec = [(0, pad_h), (0, pad_w)]  # pad bottom & right edges
            if img.ndim == 3:
                pad_spec.append((0, 0))  # keep channel dimension unchanged
            img_padded = np.pad(img, pad_spec, mode="constant", constant_values=0)
            self.imgs.append(img_padded)

        # ------------------------------------------------------------------
        # 3) Stack to numpy array and proceed with existing pipeline --------
        # ------------------------------------------------------------------
        self.imgs = np.stack(self.imgs, axis=0)
        self.get_img_dims()
        self.get_img_dims_ix()

        # get indices of additional dimensions
        self.get_ix_add_dims()
        if not image_stack:
            self.make_dropdowns_img_dims()
            ip.display.display(self.hbox_dropdowns)

    def get_img_dims(self):
        """
        Extracts height and width of an image.
        """
        img = Image.fromarray(self.imgs[0])
        self.img_height = img.height
        self.img_width = img.width

    def get_img_dims_ix(self):
        """
        Gets indices of img width and height in img shape.
        """
        self.img_shape = np.array(np.array(self.imgs).shape)

        if self.img_height != self.img_width:
            self.ix_height = np.where(self.img_shape == self.img_height)[0][0]
            self.ix_width = np.where(self.img_shape == self.img_width)[0][0]
        elif self.img_height == self.img_width:
            self.ix_height = np.where(self.img_shape == self.img_height)[0][0]
            self.ix_width = np.where(self.img_shape == self.img_width)[0][1]

    def get_ix_add_dims(self):
        """
        Gets axis of additional dimensions (number of frames and number of channels).
        """
        ix_dims = np.arange(len(np.array(self.imgs).shape))
        self.ix_diff = list(
            set(ix_dims).difference(set([self.ix_height, self.ix_width]))
        )

    def make_dropdowns_img_dims(self):
        """
        Makes dropdowns for number of channels and number of frames.
        """
        self.name_add_dims = ["num_channels", "num_images"]
        list_dropdowns = [
            self.make_dropdown(self.img_shape[ix_d], self.name_add_dims)
            for ix_d in self.ix_diff
        ]
        self.hbox_dropdowns = widgets.HBox(list_dropdowns)

    def make_dropdown(self, size_dim: int, name_add_dims: str):
        """
        Makes one dropdown per additional axis.
        :param size_dim: Length of additional axis.
        :param name_add_dims: Name of additional axis.
        """
        drop_options = name_add_dims
        dropdown = widgets.Dropdown(
            options=drop_options,
            layout=widgets.Layout(width="50%"),
            description="Dimension with length " + str(size_dim) + " is:",
            style={"description_width": "250px"},
        )
        return dropdown

    def spec_img_dims(self):
        """
        Specify image dimensions acc. to following pattern: (num_imgs, height, width, num_channels).
        """

        # get indices of frame and channel indentifier
        name_dims = [c.value for c in self.hbox_dropdowns.children]

        # check which axes are currently present and create dict for assignment of new axes
        self.dims_assign_dict = dict()
        self.axis_length_dict = dict()

        for md, ixd in zip(name_dims, self.ix_diff):
            self.dims_assign_dict[md] = ixd
            self.axis_length_dict[md] = self.img_shape[ixd]

    def align_img_dims(self):
        """
        Aligns image dimensions to following pattern: (num_imgs, height, width, num_channels).
        """

        # move axes to (num_frames, height, width, num_channels)
        self.imgs_clean = self.imgs.copy()
        self.imgs_clean = np.array(self.imgs_clean)

        if (
            "num_images" in self.dims_assign_dict.keys()
            and "num_channels" in self.dims_assign_dict.keys()
        ):
            new_ax_im_len = np.where(
                np.array(self.img_shape) == self.axis_length_dict["num_images"]
            )[0][0]
            self.imgs_clean = np.moveaxis(self.imgs_clean, new_ax_im_len, 0)

            new_ax_ch_len = np.where(
                np.array(self.imgs_clean.shape) == self.axis_length_dict["num_channels"]
            )[0][0]
            self.imgs_clean = np.moveaxis(self.imgs_clean, new_ax_ch_len, -1)

        if (
            "num_images" in self.dims_assign_dict.keys()
            and "num_channels" not in self.dims_assign_dict.keys()
        ):
            self.imgs_clean = np.moveaxis(
                self.imgs_clean, self.dims_assign_dict["num_images"], 0
            )

        if (
            "num_images" not in self.dims_assign_dict.keys()
            and "num_channels" in self.dims_assign_dict.keys()
        ):
            self.imgs_clean = np.moveaxis(
                self.imgs_clean, self.dims_assign_dict["num_channels"], -1
            )

        # add axis in case one is missing
        if "num_images" not in self.dims_assign_dict.keys():
            self.imgs_clean = np.expand_dims(self.imgs_clean, axis=0)

        if "num_channels" not in self.dims_assign_dict.keys():
            self.imgs_clean = np.expand_dims(self.imgs_clean, axis=-1)

    def show_example_image(self, img: np.ndarray):
        """
        Displays example image.
        """
        _, self.ax = plt.subplots()
        self.ax.imshow(img)
        plt.show()

    def select_channel(self):
        """
        Creates a figure linked to a dropdown to select channel.
        """

        def f(a, c):
            _, ax1 = plt.subplots()
            ax1.imshow(self.imgs_clean[int(c), :, :, int(a)])
            ax1.set_xticks([])
            ax1.set_yticks([])
            plt.title("Channel: " + str(a))
            plt.show()

        self.output_sel_ch = interactive(
            f,
            a=widgets.Dropdown(
                options=np.arange(self.imgs_clean.shape[-1]),
                layout=widgets.Layout(width="50%"),
                description="Channel",
            ),
            c=widgets.IntSlider(
                min=0,
                max=self.imgs_clean.shape[0] - 1,
                description="Image ID",
            ),
        )

    def set_channel(self):
        """
        Set selected channel based on label of dropdown.
        """
        self.selected_ch = int(self.output_sel_ch.children[0].label)
        self.imgs_sel_ch = self.imgs_clean[:, :, :, self.selected_ch]
        self.imgs_sel_ch = np.expand_dims(self.imgs_sel_ch, -1)

    def get_corners_cutout(self):
        """
        Gets axis limits for current zoom-in.
        """
        xlim = [int(xl) for xl in self.ax.get_xlim()]
        ylim = [int(yl) for yl in self.ax.get_ylim()]

        self.x_min = np.min(xlim)
        self.x_max = np.max(xlim)

        self.y_min = np.min(ylim)
        self.y_max = np.max(ylim)

    def make_cutouts(self):
        """
        Generates cutouts for all images.
        """
        self.imgs_cut = np.array(
            [
                img[self.y_min : self.y_max, self.x_min : self.x_max, 0]
                for img in self.imgs_sel_ch
            ]
        )

    def show_all_cutouts(self):
        """
        Displays all cutouts with slider.
        """

        def f(i):
            _, ax1 = plt.subplots()
            ax1.imshow(self.imgs_cut[int(i)])
            ax1.set_xticks([])
            ax1.set_yticks([])
            plt.show()

        self.output_all_cuts = interactive(
            f,
            i=widgets.IntSlider(
                min=0,
                max=self.imgs_cut.shape[0] - 1,
                description="Image ID",
            ),
        )

    def save_cutouts(self):
        """
        Saves all cutouts in new folder.
        """

        # update path to cutout images and create dir if necessary
        self.path_cut = Path(self.path_cut_base).joinpath(
            Path(self.chosen_files[0]).stem
        )
        os.makedirs(self.path_cut, exist_ok=True)

        for f, cut in zip(self.chosen_files, self.imgs_cut):
            cut_scale = self.scale_pixel_val(cut)
            io.imsave(self.path_cut.joinpath(Path(f).stem + "_cut.png"), cut_scale, check_contrast=False)
    def get_segmentation_models(self):
        """
         Collects all json files and combines model information in table.
         Ensures that the Omni default models are always listed, even if no
         corresponding weight file is found locally.
        """
        files = glob.glob(
             self.path_midap + "/model_weights/**/model_weights*.json", recursive=True
        )
 

        # ------------------------------------------------------------------
        # 1) Aggregate all JSON-based model tables (if any)
        # ------------------------------------------------------------------
        if files:
            df = pd.read_json(files[0])
            for f in files[1:]:
                df = pd.concat([df, pd.read_json(f)], axis=1)
            self.df_models = df.T
        else:
            # Fresh installation without legacy weight files
            self.df_models = pd.DataFrame()

        # ------------------------------------------------------------------
        # 2) Inject default Omni / Cellpose-Omni models
        # ------------------------------------------------------------------
        from midap.segmentation.omni_segmentator import OmniSegmentation
        from midap.segmentation.cellpose_segmentator_jupyter import CellposeSegmentationJupyter

        for mdl in OmniSegmentation.DEFAULT_MODELS:
            # Row index format must match the legacy pattern because further
            # downstream code strips the first two tokens:
            #   "model_weights_<here...>" → "<here...>"
            idx = f"model_weights_{mdl}"
            if idx in self.df_models.index:
                continue

            # Create a minimal placeholder row.  Additional columns will be
            # auto-filled with NaN and safely ignored by the GUI.
            self.df_models.loc[idx, "species"] = "OmniPose"
            self.df_models.loc[idx, "marker"] = "omni"
            self.df_models.loc[idx, "nn_type_alias"] = "OmniSegmentationJupyter"

        # --------------------------------------------------------------
        # 3) Inject default *Cellpose* models (classic, non-omni)
        # --------------------------------------------------------------
        from midap.segmentation.cellpose_segmentator_jupyter import CellposeSegmentationJupyter
        for mdl in CellposeSegmentationJupyter.DEFAULT_MODELS:
            idx = f"model_weights_{mdl}"
            if idx in self.df_models.index:
                continue

            self.df_models.loc[idx, "species"] = "Cellpose-SAM"
            self.df_models.loc[idx, "marker"]  = "cellpose-sam"
            self.df_models.loc[idx, "nn_type_alias"] = "CellposeSegmentationJupyter"



    def display_segmentation_models(self):
        """
        Shows all pretrained models in a plain three-column table

            | name | type | select |

        - name    : full model identifier (index in *df_models*)  
        - type    : the former "marker" field  
        - select : a checkbox to tick the model for further processing
        """

        # --- table header --------------------------------------------------
        header = widgets.HBox(
            [
                widgets.HTML("<b>Name</b>",  layout=widgets.Layout(width="50%")),
                widgets.HTML("<b>Type</b>",  layout=widgets.Layout(width="30%")),
                widgets.HTML("<b>Select</b>",layout=widgets.Layout(width="20%")),
            ]
        )

        # --- one row per model --------------------------------------------
        rows = []
        self.model_checkboxes = {}        # model_id → Checkbox

        for model_id, row in self.df_models.iterrows():
            lbl_name = widgets.Label(model_id, layout=widgets.Layout(width="50%"))
            lbl_type = widgets.Label(str(row.get("nn_type_alias", "")),
                                     layout=widgets.Layout(width="30%"))
            cb       = widgets.Checkbox(value=False, indent=False,
                                        layout=widgets.Layout(width="20%"))
            self.model_checkboxes[model_id] = cb
            rows.append(widgets.HBox([lbl_name, lbl_type, cb]))

        self.outp_table = widgets.VBox([header] + rows)
        display(self.outp_table)

    def select_segmentation_models(self):
        """
        Builds *self.all_chosen_seg_models* from the check-boxes created
        in *display_segmentation_models*.
        """
        self.all_chosen_seg_models = {}

        for model_id, cb in self.model_checkboxes.items():
            if cb.value:                                     # ticked
                nn_type = self.df_models.loc[model_id, "nn_type_alias"]
                self.all_chosen_seg_models.setdefault(nn_type, []).append(model_id)

    def run_all_chosen_models(self):
        """
        Runs all pretrained models of chosen model types.
        """
        self.dict_all_models = {}
        self.dict_all_models_label = {}
        for nnt, models in self.all_chosen_seg_models.items():
            self.select_segmentator(nnt)
            for model in models:
                model_name = "_".join((model).split("_")[2:])
                self.pred.run_image_stack_jupyter(
                    self.imgs_cut, model_name, clean_border=False
                )
                self.dict_all_models["{}_{}".format(nnt, model)] = self.pred.seg_bin
                self.dict_all_models_label[
                    "{}_{}".format(nnt, model)
                ] = self.pred.seg_label

                # ------------------------------------------------------
                # Free GPU memory that might still be held by the just
                # finished predictor.  This is crucial when executing
                # multiple models sequentially in the same notebook /
                # Colab runtime to avoid out-of-memory crashes.
                # ------------------------------------------------------
                if hasattr(self.pred, "cleanup"):
                    self.pred.cleanup()


    def select_and_run_segmentation_models(self,df):
        """
        Display searchable table with available models. 
        Table contents are read from a CSV file and passed into this function as a Pandas dataframe.
        A widget with a (string-searchable) list of model names allows the user to choose 2 or more models.
        'Apply selection' will select the desired models with no further action - for later use
        'Apply and run' will select desired models and do the segmentation on the images selected in earlier step
        INPUT: a dataframe containing the list of available models and their specifications
        OUTPUT: arrays of segmentation intances - masks - of each chosen model and for each selected image.
        NOTE: replaces combined functionality of original functions 'select_segmentation_models' and 'run_all_chosen_models'
        """
        self.get_segmentation_models()
        display(data_table.DataTable(df, include_index=False, num_rows_per_page=10))

        all_names = df["Model Name"].astype(str).tolist()

        search = widgets.Text(placeholder="filter models with ... (substring match)", layout=widgets.Layout(width="40%"))
        sel    = widgets.SelectMultiple(options=sorted(all_names), rows=12, description="Select")
        btn_all   = widgets.Button(description="Select all (filtered)", tootip='Select all models matching filter keywords', icon="check-square")
        btn_clear  = widgets.Button(description="Clear", tooltip='Clear selection',icon="trash")
        btn_apply = widgets.Button(description="Apply selection",tooltip='Select models for later use',icon="tasks")
        btn_applyrun   = widgets.Button(description="Apply & run", tooltip='Select models and run segmentation',icon="play-circle",button_style="primary")
        out = widgets.Output()

        def refresh_options(_=None):
            q = search.value.lower().strip()
            opts = [n for n in all_names if q in n.lower()] if q else sorted(all_names)
            current = set(sel.value)
            sel.options = opts
            sel.value = tuple([o for o in opts if o in current])

        search.observe(refresh_options, names="value")
        refresh_options()

        def on_all_clicked(_):
            sel.value = tuple(sel.options)

        def on_none_clicked(_):
            sel.value = ()

        btn_all.on_click(on_all_clicked)
        btn_clear.on_click(on_none_clicked)


        def _apply_selection(run_now=False):
            selected = set(sel.value)

            self.model_checkboxes = {
                name: widgets.Checkbox(value=(name in selected), indent=False, layout=widgets.Layout(width="1px", height="1px"))
                for name in all_names
            }

            with out:
                clear_output()
                print(f"Selected {len(selected)} model(s):")
                for n in sorted(selected):
                    print("  •", n)

            if run_now:
                self.select_segmentation_models()
                self.run_all_chosen_models_timing()

        def on_apply_clicked(_):
            _apply_selection(run_now=False)

        def on_applyrun_clicked(_):
            _apply_selection(run_now=True)

        btn_apply.on_click(on_apply_clicked)
        btn_applyrun.on_click(on_applyrun_clicked)

        sel_ui = widgets.VBox([
            widgets.HBox([search, btn_all, btn_clear, btn_apply, btn_applyrun]),
            sel,
            out
        ])

        display(sel_ui)
    
    def run_all_chosen_models_timing(self):
        """
        Runs all pretrained models of chosen model types and records inference times.
        - self.model_inference_times: seconds per image (avg over imgs_cut)
        - self.model_inference_times_total: total wall time per model (seconds)
        Returns simple table with inference times and hardware info (CPU, GPU, etc)
        NOTE: built using the original 'run_all_chosen_models' function
        """
        self.dict_all_models = {}
        self.dict_all_models_label = {}
        self.model_inference_times = {}
        self.model_inference_times_total = {}

        try:
            n_imgs = len(self.imgs_cut)
        except Exception:
            n_imgs = 1

        rows = []

        for nnt, models in self.all_chosen_seg_models.items():
            self.select_segmentator(nnt)
            for model in models:
                model_name = "_".join((model).split("_")[2:])

                key = f"{nnt}_{model}"
                t0 = time.perf_counter()

                self.pred.run_image_stack_jupyter(
                    self.imgs_cut, model_name, clean_border=False
                )
                elapsed = time.perf_counter() - t0

                self.dict_all_models[key] = self.pred.seg_bin
                self.dict_all_models_label[key] = self.pred.seg_label

                self.model_inference_times_total[key] = elapsed
                self.model_inference_times[key] = elapsed / max(1, n_imgs)

                # ---- time inference summary table ---------
                rows.append({
                    "Model": key,
                    "Images": n_imgs,
                    "Total time (s)": elapsed,
                    "Images / s": (n_imgs / elapsed) if elapsed > 0 else float("inf"),
                })

                # ------------------------------------------------------
                # Free GPU memory that might still be held by the just
                # finished predictor.  This is crucial when executing
                # multiple models sequentially in the same notebook /
                # Colab runtime to avoid out-of-memory crashes.
                # ------------------------------------------------------
                if hasattr(self.pred, "cleanup"):
                    self.pred.cleanup()
    


        print("\n\n\n\n\n==== Inference Time Summary ====\n")
        if rows:
            self.print_runtime_env()
            df = pd.DataFrame(rows)
            display(df)


    def print_runtime_env(self):
        """Print a one-line summary of the compute device (TPU/GPU/CPU)."""
        # TPU
        if os.environ.get("COLAB_TPU_ADDR"):
            print("____ Running on TPU ____")
            return

        # GPU
        try:
            gpu_name = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            if gpu_name:
                print(f"____ Running on GPU: {gpu_name.splitlines()[0]} ____")
                return
        except Exception:
            pass

        # CPU 
        try:
            cpu = subprocess.check_output(
                "lscpu | grep 'Model name' | sed 's/.*: //'",
                shell=True, text=True
            ).strip()
            if cpu:
                print(f"____ Running on CPU: {cpu} ____")
                return
        except Exception:
            pass

        print("____ Running on CPU ____")

    
    
    def select_segmentator(self, segmentation_class: str):
        """
        Selects segmentator based on segmentation class.
        :param segmentation_class: Name of segmentation class.
        """
        if segmentation_class == "OmniSegmentationJupyter":
            path_model_weights = Path(self.path_midap).joinpath(
                "model_weights", "model_weights_omni"
            )
        else:
            path_model_weights = Path(self.path_midap).joinpath(
                "model_weights", "model_weights_legacy"
            )

        # define variables
        postprocessing = False
        network_name = None
        img_threshold = 255


        # get the right subclass
        class_instance = None

        segmentation_subclasses = get_inheritors(base_segmentator.SegmentationPredictor)
        jupyter_seg_cls = [s for s in segmentation_subclasses if "Jupyter" in s.supported_setups]
        
        #for subclass in get_inheritors(base_segmentator.SegmentationPredictor):
        for subclass in jupyter_seg_cls:
            if subclass.__name__ == segmentation_class:
                class_instance = subclass

        # throw an error if we did not find anything
        if class_instance is None:
            raise ValueError(f"Chosen class does not exist: {segmentation_class}")

        # make sure GPU memory held by a previous predictor is freed
        if hasattr(self, "pred") and hasattr(self.pred, "cleanup"):
            self.pred.cleanup()

        # get the Predictor
        self.pred = class_instance(
            path_model_weights=path_model_weights,
            postprocessing=postprocessing,
            model_weights=network_name,
            img_threshold=img_threshold,
        )

    # ------------------------------------------------------------------
    #  Quantitative agreement analysis
    # ------------------------------------------------------------------
    def compute_model_diff_scores(self):
        """
        Calculates a mean and standard deviation of semantic-segmentation
        disagreement scores for every model that has been run.

        Step 1 – pairwise disagreement:
            For every unordered model pair (m1, m2) the fraction of pixels
            whose semantic class labels differ is computed for each image
            and averaged over the whole image stack.

        Step 2 – per-model aggregation:
            A model's final score is the mean of the pairwise scores of
            all pairs that include this very model.

        Returns
        -------
        dict
            model_id → (mean disagreement, std deviation)
        """
        models = list(self.dict_all_models.keys())
        if len(models) < 2:
            raise RuntimeError("Need at least two models to compare.")

        # ---- pair-wise disagreement -----------------------------------
        diff_pair = {}                                   # (m1,m2) → score
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                sem_a, sem_b = self.dict_all_models[m1], self.dict_all_models[m2]

                per_img = []
                for a_img, b_img in zip(sem_a, sem_b):
                    a_img = np.asarray(a_img)
                    b_img = np.asarray(b_img)
                    if a_img.ndim == 3 and a_img.shape[-1] == 2:
                        a_img = a_img[..., 0]
                    if b_img.ndim == 3 and b_img.shape[-1] == 2:
                        b_img = b_img[..., 0]

                    per_img.append((a_img != b_img).mean())

                diff_pair[(m1, m2)] = float(np.mean(per_img))

        # ---- aggregate per model --------------------------------------
        model_vals = {m: [] for m in models}
        for (m1, m2), val in diff_pair.items():
            model_vals[m1].append(val)
            model_vals[m2].append(val)

        return {m: (float(np.mean(v)), float(np.std(v))) for m, v in model_vals.items()}

    def compare_segmentations(self):
        """
        Visualises:
          1. raw image
          2. instance segmentation of model-1
          3. instance segmentation of model-2
          4. bar-plot of the per-model mean semantic disagreement scores
             with standard deviation as error bars.
        """

        # ----------------------------------------------------------------
        # prepare bar-plot data (only once)
        # ----------------------------------------------------------------
        if not hasattr(self, "model_diff_scores"):
            self.model_diff_scores = self.compute_model_diff_scores()

        def f(a, b, c):
            fig = plt.figure(figsize=(20, 6))

            # --- raw image ---------------------------------------------
            raw = self.imgs_cut[int(c)]
            ax0 = fig.add_subplot(141)
            ax0.imshow(raw, cmap="gray")
            ax0.set_xticks([]); ax0.set_yticks([])
            ax0.set_title("Raw image")

            # --- instance seg – model 1 --------------------------------
            inst_a = self.dict_all_models_label[a][int(c)]
            inst_a = np.asarray(inst_a)
            if inst_a.ndim == 3 and inst_a.shape[-1] == 2:
                inst_a = inst_a[..., 0]
            inst_a = np.ma.masked_where(inst_a == 0, inst_a)

            ax1 = fig.add_subplot(142, sharex=ax0, sharey=ax0)
            ax1.imshow(inst_a, cmap="tab20")
            ax1.set_xticks([]); ax1.set_yticks([])
            ax1.set_title("Model 1 (instance)")

            # --- instance seg – model 2 --------------------------------
            inst_b = self.dict_all_models_label[b][int(c)]
            inst_b = np.asarray(inst_b)
            if inst_b.ndim == 3 and inst_b.shape[-1] == 2:
                inst_b = inst_b[..., 0]
            inst_b = np.ma.masked_where(inst_b == 0, inst_b)

            ax2 = fig.add_subplot(143, sharex=ax0, sharey=ax0)
            ax2.imshow(inst_b, cmap="tab20")
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.set_title("Model 2 (instance)")

            # --- bar-plot with mean disagreements and std dev ---------
            ax3 = fig.add_subplot(144)
            mdl_ids = list(self.model_diff_scores.keys())
            scores, std_devs = zip(*[self.model_diff_scores[m] for m in mdl_ids])
            # Shorten model names
            short_mdl_ids = [
                f"{m[:5]}...{m.split('_')[-1]}" for m in mdl_ids
            ]
            ax3.bar(range(len(mdl_ids)), scores, yerr=std_devs, color="steelblue", capsize=5)
            ax3.set_xticks(range(len(mdl_ids)))
            ax3.set_xticklabels(short_mdl_ids, rotation=90)
            ax3.set_ylabel("Mean semantic difference")
            ax3.set_title("Per-model disagreement")

            plt.tight_layout()
            plt.show()

        self.output_seg_comp = interactive(
            f,
            a=widgets.Dropdown(
                options=self.dict_all_models.keys(),
                description="Model 1", layout=widgets.Layout(width="45%")
            ),
            b=widgets.Dropdown(
                options=self.dict_all_models.keys(),
                description="Model 2", layout=widgets.Layout(width="45%")
            ),
            c=widgets.IntSlider(
                min=0,
                max=len(next(iter(self.dict_all_models.values()))) - 1,
                description="Image ID"
            ),
        )
        display(self.output_seg_comp)

    def display_buttons_weights(self):
        """
        Displays all used models for segmentation to select best model.
        """
        self.out_weights = widgets.RadioButtons(
            options=list(self.dict_all_models.keys()),
            description="Model weights:",
            disabled=False,
            layout=widgets.Layout(width="100%"),
        )

    def load_add_files(self):
        """
        Loads additional (full) image stack.
        """

        def f(a):
            if a == True:
                self.fc_add_file = FileChooser(self.path)
                self.fc_add_file.show_only_dirs = True
                self.fc_add_file.layout ={"width": "600px"}
                ip.display.display(self.fc_add_file)

        self.out_add_file = interactive(
            f,
            a=widgets.Checkbox(
                value=False,
                description="Do you want to select an additional dataset for the segmentation?",
            ),
        )

    def segment_all_images(self, model_name: str):
        """
        Segments all images for given model type and selected model weights.
        :param model_name: Name of chosen trained model.
        """
        self.pred.run_image_stack_jupyter(self.imgs_cut, model_name, clean_border=False)

        # free GPU memory right after a single-model inference run
        if hasattr(self.pred, "cleanup"):
            self.pred.cleanup()

    def process_images(self):
        """
        Processes all images after loading the full image stack.
        """
        if self.out_add_file.children[0].value == True:
            self.chosen_files = os.listdir(self.fc_add_file.selected)
            self.chosen_dir = self.fc_add_file.selected
            self.load_input_image(image_stack=True)
            #self.get_img_dims_ix()
            #self.spec_img_dims()
            self.align_img_dims()
            self.set_channel()
            self.make_cutouts()
            self.save_cutouts()

        self.select_segmentator(self.out_weights.label.split("_")[0])
        self.segment_all_images(("_").join(self.out_weights.label.split("_")[3:]))
        self.save_segs()

    def save_segs(self):
        """
        Saves all segmentations in new folder.
        """

        self.path_seg = Path(self.path_seg_base).joinpath(
            Path(self.chosen_files[0]).stem
        )
        os.makedirs(self.path_seg, exist_ok=True)

        segs = np.array(self.pred.seg_label)

        for f, seg in zip(self.chosen_files, segs):
            io.imsave(self.path_seg.joinpath(Path(f).stem + "_seg.tif"), seg, check_contrast=False)

    def get_usern_pw(self):
        """
        Get username and password for upload to polybox.
        """

        self.out_usern = Text(
            value="", placeholder="", description="Username:", disabled=False
        )

        self.out_passw = Password(
            value="", placeholder="", description="Password:", disabled=False
        )

        self.button = Button(description="Confirm")
        self.output = Output()

        def on_button_clicked(b):
            with self.output:
                arg1 = self.out_usern.value
                arg2 = self.out_passw.value
                arg3 = str(self.path_seg).rstrip("/")
                subprocess.call(
                    "./upload_polybox.sh "
                    + str(arg1)
                    + " "
                    + str(arg2)
                    + " "
                    + str(arg3),
                    shell=True,
                )

        self.button.on_click(on_button_clicked)

    def scale_pixel_val(self, img):
        """
        Rescale the pixel values of the image
        :param img: The input image as array
        :returns: The images with pixels scales to standard RGB values
        """
        img_scaled = (255 * ((img - np.min(img)) / np.max(img - np.min(img)))).astype(
            "uint8"
        )
        return img_scaled
