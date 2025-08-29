import argparse
import os
import sys
from pathlib import Path
from shutil import rmtree, unpack_archive
from typing import Union, Optional

import requests
from tqdm import tqdm
import json


def query_yes_no(question: str, default="yes"):
    """
    Ask a yes/no question via raw_input() and return their answer.
    :param question: a string that is presented to the user.
    :param default: Is the presumed answer if the user just hits <Enter>. It must be "yes" (the default),
                    "no" or None (meaning an answer is required of the user).
    :return: The "answer" return value is True for "yes" or False for "no".
    """

    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def download_file(
    url: str, fname: Union[str, bytes, os.PathLike], desc: Optional[str] = None
):
    """
    Downloads a file as stream (not full file in memory)
    :param url: The URL of the file
    :param fname: The name of the file
    :param desc: The prefix used for the download progress bar
    """

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fname, "wb") as f:
            pbar = tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=desc)
            pbar.clear()
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)
            pbar.close()


def main(args=None):
    """
    Downloads the files necessary for the midap pipeline
    :param args: The args used for parsing
    """

    # get the args
    if args is None:
        args = sys.argv[1:]

    # arg parsing
    parser = argparse.ArgumentParser(
        description="Download the files for the MIDAP pipeline."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Option to force a download even though files already " "exist.",
    )
    args = parser.parse_args(args)

    # this file indicates a succesful download
    done_file = ".success"

    # the path to the root of the repo
    root = Path(__file__).parent.parent.parent

    # download the files
    with open(Path(__file__).parent.joinpath("download_info.json"), "r") as f:
        d_dict = json.load(f)
    downloads = [
        (
            d_dict["model_weights"]["url"],
            d_dict["model_weights"]["name"],
            d_dict["model_weights"]["version"],
        ),
      
    ]
    for url, fname, version in downloads:
        # The full path of the downloaded file and the folder of the unpacked file
        zip_file = root.joinpath(fname)
        final_folder = Path(os.path.splitext(zip_file)[0])
        current_done = final_folder.joinpath(done_file)

        # Download the file
        try:
            # check if the current files are up to date
            up_to_date = False
            if current_done.exists():
                with open(current_done, "r") as f:
                    content = f.read()
                if content == version:
                    up_to_date = True
                else:
                    print(f"New version of {zip_file.name} available!")

            if up_to_date:
                # ask to download again if we want to
                if args.force:
                    answer = query_yes_no(
                        f"{zip_file.name} appears to be already downloaded, overwrite?"
                    )
                    if not answer:
                        continue
                else:
                    continue

            # download
            download_file(url=url, fname=zip_file, desc=f"Downloading {fname}")

            # remove the old folder
            if final_folder.exists():
                rmtree(final_folder, ignore_errors=False)

            # unzip
            unpack_archive(filename=zip_file, extract_dir=zip_file.parent)

            # create the donefile
            with open(current_done, "w+") as f:
                f.write(version)
        finally:
            # clean up if necessary
            zip_file.unlink(missing_ok=True)
