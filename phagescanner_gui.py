"""
PhageScanner GUI allows for visualization of predicted proteins for each ORF.
It also allows for obtaining all proteins within a genomic window, thereby 
retrieving all proteins of interest. The only dependency is an output
predictions file from using the `predict` pipeline in PhageScanner.

Use this for info: python phagescanner_gui.py -h
"""
import argparse
from enum import Enum
from pathlib import Path
import sys
from PhageScanner.gui import feature_plotter
import tkinter as tk
from PhageScanner.gui.gui_frames import Application



class GUIOptions(Enum):
    """Names of options for the gui application.

    Description:
        This enum contains the options for the gui application.
        The GUI depends on png files generated from genomes,
        so this is the options that must be used first.
    """

    create_images = "create_images"
    gui = "gui"

def parseArgs(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        help="Choose which pipeline to run", dest="sub_parser"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true")

    # if create images
    create_images_parser = subparsers.add_parser(
        GUIOptions.create_images.value, help="Create images for the GUI. Must be ran first."
    )
    create_images_parser.add_argument(
        "-p",
        "--prediction_csv",
        type=Path,
        help="path to the phagescanner prediction csv file.",
        required=True,
    )
    create_images_parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="path for storing the images. (MUST BE EMPTY or nonexistent)",
        required=True,
    )

    # if gui
    train_pipeline_parser = subparsers.add_parser(
        GUIOptions.gui.value, help="Open a GUI for exploring the genomes/contigs."
    )
    train_pipeline_parser.add_argument(
        "-p",
        "--prediction_csv",
        type=Path,
        help="path to the phagescanner prediction csv file.",
        required=True,
    )
    train_pipeline_parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="path for storing the images. (MUST BE EMPTY or nonexistent)",
        required=True,
    )

    return parser.parse_args(argv)

def main():
    def create_images():
        print(f"Creating images from data...")
        print(f"(Takes time!)")
        feature_plotter.create_genome_images(path_to_predictions=args.prediction_csv,
                                             output_path=args.out)
        
    def run_gui():
        print(f"Starting the GUI...")
        root = tk.Tk()
        root.title('PhageScanner Genome Explorer')
        root.geometry('800x685')
        app = Application(master=root,
                          image_dir=args.out,
                          prediction_csv=args.prediction_csv)
        app.mainloop()

    # arguments
    args = parseArgs(sys.argv[1:])

    # map subparsers to their corresponding functions and messages
    actions = {
        GUIOptions.create_images.value: create_images,
        GUIOptions.gui.value: run_gui,
    }

    # run benchmark type specified
    action = actions.get(args.sub_parser)
    if action:
        action()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
