import base64
import io
import logging
import os
import sys
import threading
from typing import Collection, Union, Tuple, Optional

import PIL
import midap.apps.PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_logger(filepath, logging_level=7):
    """
    Get logger, from given file path and with logging level
    :param filepath: name of the file that is calling the logger, used to give it a name.
    :param logging_level: A number from 0 to 7 indicating the amount of output, defaults to 7 (debug output)
    :return: logger object
    """

    logger = logging.getLogger(os.path.basename(filepath))

    if len(logger.handlers) == 0:
        log_formatter = logging.Formatter(
            fmt="%(asctime)s %(name)10s %(levelname).3s   %(message)s ",
            datefmt="%y-%m-%d %H:%M:%S",
            style="%",
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_formatter)
        logger.addHandler(stream_handler)
        logger.propagate = False
        set_logger_level(logger, logging_level)

    return logger


def set_logger_level(logger, level):
    """
    Sets the level of a logger
    :param logger: The logger object to set the level
    :param level: The level, a number from 0 to 7 (corresponding to the log level in the bash script)
    """

    if level <= 2:
        log_level = logging.CRITICAL
    elif level == 3:
        log_level = logging.ERROR
    elif level == 4:
        log_level = logging.WARNING
    elif level <= 6:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    logger.setLevel(log_level)


def get_inheritors(klass):
    """
    Get all child classes of a given class
    :param klass: The class to get all children
    :return: All children as a set
    """

    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


def convert_to_bytes(
    file_or_bytes: Union[str, bytes], resize: Optional[Tuple[int, int]] = None
):
    """
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :param resize:  optional new size
    :return: (bytes) a byte-string object
    """

    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize(
            (int(cur_width * scale), int(cur_height * scale)), PIL.Image.ANTIALIAS
        )
    with io.BytesIO() as bio:
        img.save(bio, format="GIF")
        del img
        return bio.getvalue()


def GUI_selector(
    figures: Collection[plt.Figure], labels: Collection[str], title="", close_figs=True
):
    """
    Starts up a GUI selector for imgs and labels
    :param figures: A list of figures that will presented in the GUI as buttons that the user can select
    :param labels: A list of labels corresponding to the input images
    :param title: Title for the GUI
    :param close_figs: Close all figures after the GUI has extracted the data
    :return: The label that the user selected by clicking on the corresponding image
    """

    # check
    if len(figures) != len(labels):
        raise ValueError("Number of figures does not math number of labels!")

    # get the number of cols for the layout
    num_cols = int(np.ceil(np.sqrt(len(labels))))

    # create buttons for the GUI
    buffers = []
    buttons = []
    new_line = []
    for i, (fig, label) in enumerate(zip(figures, labels)):
        # figure to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        buf = convert_to_bytes(buf.read())
        buffers.append(buf)
        # the first button starts as selected
        if i == 0:
            new_line.append(
                sg.Button(
                    "",
                    image_data=buf,
                    button_color=("black", "yellow"),
                    border_width=5,
                    key=label,
                )
            )
            marked = label
        else:
            new_line.append(
                sg.Button(
                    "",
                    image_data=buf,
                    button_color=(
                        sg.theme_background_color(),
                        sg.theme_background_color(),
                    ),
                    border_width=5,
                    key=label,
                )
            )
        if len(new_line) == num_cols:
            buttons.append(new_line)
            new_line = []

    # if we still have elements in the line, append the line
    if len(new_line) > 0:
        buttons.append(new_line)

    # get the number of rows
    num_rows = len(buttons)

    # close all figs
    if close_figs:
        plt.close("all")

    # The GUI
    layout = buttons
    layout += [[sg.Column([[sg.OK(), sg.Cancel()]], key="col_final")]]
    window = sg.Window(
        title, layout, element_justification="c", resizable=True, finalize=True
    )
    old_window_size = window.size
    window.bind("<Configure>", "-CONFIG-")

    # Time to wait before resizing windows
    THREADWAIT = 0.1
    # threshold to increase or decrease window size before resizing the elements
    INCWIN = 100
    REDWIN = 100

    def resize_buttons():
        """
        Resized the buttons of the window to fit new window size
        """
        # Get the new window size
        win_h, win_w = window.size
        # reduce width to fit the other buttons
        win_w -= 120
        # resize the imgage
        for buf, label in zip(buffers, labels):
            window[label].update(
                image_data=convert_to_bytes(
                    buf, ((win_h / num_rows), (win_w / num_cols))
                )
            )

    # timer for the resize
    timer_windowResize = threading.Timer(THREADWAIT, resize_buttons)

    # Event Loop
    while True:
        # Read event
        event, values = window.read()
        # break if we have one of these
        if event in (sg.WIN_CLOSED, "Exit", "Cancel", "OK"):
            break

        # get the last event
        for i, l in enumerate(labels):
            # if the last event was an image button click, mark it
            if event == l:
                marked = l
                break
        # maked button is highlighted
        for l in labels:
            if marked == l:
                window[l].update(button_color=("black", "yellow"))
            else:
                window[l].update(
                    button_color=(
                        sg.theme_background_color(),
                        sg.theme_background_color(),
                    )
                )

        # config event
        if event == "-CONFIG-":
            # check if the size change was enough to trigger a resize
            if (
                ((old_window_size[0] + INCWIN) < window.size[0])
                or ((old_window_size[0] - REDWIN) > window.size[0])
                or ((old_window_size[1] + INCWIN) < window.size[1])
                or ((old_window_size[1] - REDWIN) > window.size[1])
            ):
                # The idea behind the logic below is that we cancel a time each time the size threshold is reached
                # and then restart it. This means we only resize the element after the user is done with his
                # resize action because during the resize action the timer is constantly cancelled

                # We cancel the time
                timer_windowResize.cancel()
                # we set the time
                timer_windowResize = threading.Timer(THREADWAIT, resize_buttons)
                # (Re)start the timer to resize the image after THREADWAIT-amount of seconds
                timer_windowResize.start()

                # set the new window size
                old_window_size = window.size

    window.close()

    if event != "OK":
        raise InterruptedError("GUI was cancelled or unexpectedly closed, exiting...")

    return marked
