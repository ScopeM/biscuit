import argparse
import os

from typing import Union
from pathlib import Path

# to get all subclasses
from midap.segmentation import *
from midap.segmentation import base_segmentator
from midap.utils import get_inheritors

### Functions
#############


def main(
    path_model_weights: Union[str, bytes, os.PathLike],
    path_pos: Union[str, bytes, os.PathLike],
    path_channel: str,
    segmentation_class: str,
    postprocessing: bool,
    clean_border: bool,
    network_name: Union[str, bytes, os.PathLike, None] = None,
    just_select=False,
    img_threshold=1.0,
):
    """
    Performs cell segmentation on all images in a given directory
    :param path_model_weights: The path to the pretrained model weights
    :param path_pos: The path to the current identifier, the base directory for all data
    :param path_channel: The name of the current channel
    :param segmentation_class: The name of the segmentation class to use
    :param postprocessing: whether to use postprocessing or not
    :param clean_border: whether to clean border or not
    :param network_name: Optional name of the network to skip interactive selection
    :param just_select: If True, just the network selection is performed
    :param img_threshold: The threshold for the image to cap large values of the pixels
    :return: The name of the selected model weights, note that if just_select is True and the model weights are provided
             a check is performed if the model class actually exists and the model weights are returned if so
    """

    # get the right subclass
    class_instance = None
    for subclass in get_inheritors(base_segmentator.SegmentationPredictor):
        if subclass.__name__ == segmentation_class:
            class_instance = subclass

    # throw an error if we did not find anything
    if class_instance is None:
        raise ValueError(f"Chosen class does not exist: {segmentation_class}")

    # get the Predictor
    pred = class_instance(
        path_model_weights=path_model_weights,
        postprocessing=postprocessing,
        model_weights=network_name,
        img_threshold=img_threshold,
    )

    # set the paths
    path_channel = Path(path_pos).joinpath(path_channel)
    # TODO this should not be hardcoded
    path_cut = path_channel.joinpath("cut_im")
    path_cut.mkdir(exist_ok=True)

    # now we select the segmentor
    pred.set_segmentation_method(path_cut)
    # make sure that if this is a path, we have it absolute
    if (
        pred.model_weights is not None
        and (weight_path := Path(pred.model_weights).absolute()).exists()
    ):
        pred.model_weights = str(weight_path)

    # if we just want to set the method we are done here
    if just_select:
        return pred.model_weights

    # run the stack if we want to
    pred.run_image_stack(path_channel, clean_border)
    return pred.model_weights


# Main
######

if __name__ == "__main__":
    # arg parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_model_weights",
        type=str,
        required=True,
        help="Path to the model weights that will be used " "for the segmentation.",
    )
    parser.add_argument(
        "--path_pos",
        type=str,
        required=True,
        help="Path to the current identifier folder to work on.",
    )
    parser.add_argument(
        "--path_channel",
        type=str,
        required=True,
        help="Name of the current channel to process.",
    )
    parser.add_argument(
        "--segmentation_class",
        type=str,
        help="Name of the class used for the cell segmentation. Must be defined in a file of "
        "midap.segmentation and a subclass of midap.segmentation.SegmentationPredictor",
    )
    parser.add_argument(
        "--postprocessing", action="store_true", help="Flag for postprocessing."
    )
    args = parser.parse_args()

    # run
    main(**vars(args))
