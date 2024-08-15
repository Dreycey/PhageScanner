"""Module for creating the GUI.

Description:
    This module uses tkinter to create a GUI
    for visually assessing the predictions
    on the input contigs or genomes within the
    PhageScanner ML pipeline.
"""

import os
import tkinter as tk
from pathlib import Path
from tkinter import Canvas, Scrollbar
from typing import List

import pandas as pd
from PIL import Image, ImageTk

from PhageScanner.gui import feature_plotter
from PhageScanner.main.exceptions import PipelineExceptionError
from PhageScanner.main.utils import CSVUtils


class LogoFrame(tk.Frame):
    """LogoFrame - for displaying the logo."""

    def __init__(self, master=None):
        """Create a LogoFrame.

        Description:
            Holds the Logo - of course!
        """
        super().__init__(master)
        self.pack(fill="both", expand=True, padx=20, pady=0)

        # open an image file
        logo_image = Image.open("misc/logo.png")  # replace with your image file path

        # resize the image w/o changing height-width ratio
        desired_height = 100
        ratio = desired_height / logo_image.height
        desired_width = int(logo_image.width * ratio)
        logo_image = logo_image.resize((desired_width, desired_height), Image.LANCZOS)

        # convert image for tkinter
        self.logo_image_tk = ImageTk.PhotoImage(logo_image)

        # label for the logo
        self.logo_label = tk.Label(self, image=self.logo_image_tk)
        self.logo_label.image = self.logo_image_tk

        # place the label in the top left corner of the frame
        self.logo_label.pack(anchor="center")  # 'nw' means top left

        self.start_label = tk.Label(
            self, text="Genome Explorer", font=("Helvetica", 30, "italic")
        )
        self.start_label.pack(side="top", padx=10)


class GenomeFrame(tk.Frame):
    """GenomeFrame - for displaying the genome/contig and features."""

    def __init__(self, master=None, init_image=None):
        """Create a GenomeFrame.

        Description:
            The GenomeFrame contains the image
            of the current genome or contig
            being observed.
        """
        super().__init__(master, bg="green")
        self.pack(fill="both", expand=True, padx=20, pady=0)

        # set desired height - saved as attribute for later use
        self.desired_height = 300

        # create a canvas for image - same height as image
        self.canvas = Canvas(self, width=8000, height=self.desired_height)
        self.canvas.pack(side="left", fill="both", expand=True)

        # create initial image
        self.update_image(init_image)

    def update_image(self, image: Path):
        """Update the image displayed on the screen."""
        # open the image file
        img = Image.open(image)

        # resize the image w/o changing height-width ratio
        ratio = self.desired_height / img.height
        desired_width = int(img.width * ratio)
        img = img.resize((desired_width, self.desired_height), Image.LANCZOS)

        # convert image for tkinter
        self.imgtk = ImageTk.PhotoImage(img)

        # add the image to the canvas
        self.img_id = self.canvas.create_image(0, 0, image=self.imgtk, anchor="nw")
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        # store image to avoid garbage collection
        self.canvas.image = self.imgtk


class ScrollFrame(tk.Frame):
    """ScrollFrame - for allowing scrolling the genome/contig."""

    def __init__(self, master=None, canvas=None):
        """Create a ScrollFrame.

        Description:
            The ScrollFrame allows for scrolling
            over the genomic feature image.
        """
        super().__init__(master, bg="black", height=150, width=800)
        self.grid_columnconfigure(0, weight=1)  # allow column to grow
        self.pack(fill="both", expand=False, padx=20, pady=0)

        # scrollbar
        self.scrollbar = Scrollbar(self, orient="horizontal")
        self.scrollbar.pack(fill="x")

        # put the scrollbar within the canvas
        if canvas:
            canvas.config(xscrollcommand=self.scrollbar.set)
            self.scrollbar.config(command=canvas.xview)


class ButtonFrame(tk.Frame):
    """ButtonFrame - for traversing different contigs or genomes."""

    def __init__(self, master=None, canvas=None, image_set=None):
        """Create a ButtonFrame.

        Description:
            The ButtonFrame contains the buttons
            for navigating through different contigs
            and genomes.
        """
        super().__init__(master, height=100, width=800)
        self.pack(fill="both", expand=True)

        # path to image set
        self.image_set = image_set
        self.image_index = 0

        # create pointer to genome canvase
        self.canvas = canvas

        # center the navigation buttons
        self.button_container = tk.Frame(self)
        self.button_container.place(relx=0.5, rely=0.5, anchor="center")

        # label for navigation
        self.label = tk.Label(
            self.button_container,
            text="Navigate contigs/genomes",
            fg="black",
            font=("Helvetica", 20),
        )
        self.label.pack()  # place it above the buttons

        # 'Back' button
        self.back_button = tk.Button(
            self.button_container, text="Back", command=self.back_button_pressed
        )
        self.back_button.pack(side="left", padx=10, pady=10)

        # 'Next' button
        self.next_button = tk.Button(
            self.button_container, text="Next", command=self.next_button_pressed
        )
        self.next_button.pack(side="right", padx=10, pady=10)

    def back_button_pressed(self):
        """Go back to previous contig or genome."""
        if self.image_index > 0:
            self.back_button.config(bg="blue", activebackground="blue")
            self.image_index -= 1
            self.canvas.update_image(self.image_set[self.image_index])
        else:
            self.back_button.config(bg="red", activebackground="red")

    def next_button_pressed(self):
        """Go to next contig or genome."""
        if self.image_index < len(self.image_set):
            self.back_button.config(bg="blue", activebackground="blue")
            self.image_index += 1
            self.canvas.update_image(self.image_set[self.image_index])
        else:
            self.back_button.config(bg="red", activebackground="red")

    def get_current_accession(self):
        """Return the current accession id."""
        file_name = self.image_set[self.image_index]
        return file_name.stem


class SequenceFrame(tk.Frame):
    """SequenceFrame - for obtaining proteins from a range."""

    def __init__(
        self,
        master=None,
        button_frame: ButtonFrame = None,
        prediction_df: pd.DataFrame = None,
    ):
        """Create a SequenceFrame.

        Description:
            The SequenceFrame allows for obtaining sequences
            between a specified range on the genome or
            contig being observed.
        """
        super().__init__(master, height=150, width=800)
        self.pack(fill="both", expand=True)

        # button frame
        self.button_frame: ButtonFrame = button_frame

        # dataframe
        self.prediction_df = prediction_df

        # create a new frame for the button and text boxes
        self.top_container = tk.Frame(self)
        self.top_container.pack(padx=1)  # add some vertical padding

        # 'Get Sequence' button
        self.submit_button = tk.Button(
            self.top_container,
            command=self.display_sequence,
            text="Get Sequence",
            activebackground="red",
        )
        self.submit_button.pack(side="left", padx=10)

        # 'Start' label and text box
        self.start_label = tk.Label(
            self.top_container, text="Start:", font=("Helvetica", 15)
        )
        self.start_label.pack(side="left", padx=10)
        self.start_entry = tk.Entry(self.top_container, bg="white")
        self.start_entry.pack(side="left", padx=10)

        # 'End' label and text box
        self.end_label = tk.Label(
            self.top_container, text="End:", font=("Helvetica", 15)
        )
        self.end_label.pack(side="left", padx=10)
        self.end_entry = tk.Entry(self.top_container, bg="white")
        self.end_entry.pack(side="left", padx=10)

        # frame for the large text box
        self.bottom_container = tk.Frame(self)
        self.bottom_container.pack(pady=10)  # add some vertical padding

        # add large text box
        self.text_area = tk.Text(self.bottom_container, height=10, width=400)
        self.text_area.pack(padx=10)

        # prepopulate the large text box
        self.text_area.insert("1.0", "GET DESIRED PROTEINS HERE!")

    def display_sequence(self):
        """Change the sequence displayed in the GUI."""
        # obtain the text from the 'start' and 'end' text boxes
        start_text = self.start_entry.get()
        end_text = self.end_entry.get()

        # make sure start and end can be turned into integers
        if not start_text.isdigit() or not end_text.isdigit():
            self.text_area.delete("1.0", "end")  # clear existing text
            self.text_area.insert(
                "1.0", "start/end must be integers!!"
            )  # insert the combined text
            return None

        # get subdataframe for accession
        current_accesion = self.button_frame.get_current_accession()
        accession_df = self.prediction_df[
            self.prediction_df["accession"] == current_accesion
        ]

        # check if accession isn't found
        if len(accession_df) == 0:
            accession_set = self.prediction_df["accession"].unique()
            message = f"Accession wasn't found.. {current_accesion}. \n\n"
            message += f"Accessions: {accession_set}"
            self.text_area.delete("1.0", "end")  # clear existing text
            self.text_area.insert("1.0", message)  # insert the combined text
            return None

        # select for proteins between start and end
        accession_df = accession_df[accession_df["start_pos"] >= int(start_text)]
        accession_df = accession_df[accession_df["stop_pos"] <= int(end_text)]

        # create string of all proteins within range
        feature_names = feature_plotter.get_feature_names(accession_df)
        out_string = []
        for _, row in accession_df.iterrows():
            name = ">"
            for feature in feature_names:
                if feature == "ORF":
                    continue  # TODO: clean this up.
                name += row[feature] + " "
            out_string.append(name)
            out_string.append(row["protein"])

        # display sequences in the large text box
        self.text_area.delete("1.0", "end")  # clear existing text
        self.text_area.insert("1.0", "\n".join(out_string))  # insert the combined text


def helper_get_images(image_dir: Path) -> List[Path]:
    """Return a list of image paths."""
    image_paths = []
    for file in os.listdir(image_dir):
        if file.endswith(".png"):
            image_paths.append(image_dir / file)

    if len(image_paths) == 0:
        error_msg = "No images found! Are you sure this is the correct path? "
        error_msg += (
            "Make sure to run the `python phagescanner create_images` command first."
        )
        raise PipelineExceptionError(error_msg)

    return image_paths


class Application(tk.Frame):
    """Application - creates the GUI."""

    def __init__(self, image_dir: Path, prediction_csv: Path, master=None):
        """Construct the Application GUI.

        Description:
            This constructor creates the Application, which contains
            frames and information about the prediction.
        """
        super().__init__(master)
        self.master = master
        self.pack(fill="both", expand=False)

        # get a list of image paths
        image_paths: List[Path] = helper_get_images(image_dir)

        # turn prediction csv into dataframe
        prediction_df: pd.DataFrame = CSVUtils.csv_to_dataframe(prediction_csv)

        # frames
        self.logo_frame = LogoFrame(self)
        self.genome_frame = GenomeFrame(self, init_image=image_paths[0])
        self.scroll_frame = ScrollFrame(self, self.genome_frame.canvas)
        self.button_frame = ButtonFrame(self, self.genome_frame, image_set=image_paths)
        self.sequence_frame = SequenceFrame(
            self, self.button_frame, prediction_df=prediction_df
        )
