""" Module for creating genomic/contig feature images.

Description:
    This module takes in a prediction csv (output from the pipeline)
    and creates images for each contig or genome with the predicted
    features along the genome/contig.

NOTE:
    This module heavily relies on `dna_features_viewer`.
    https://edinburgh-genome-foundry.github.io/DnaFeaturesViewer/
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from dna_features_viewer import GraphicFeature, GraphicRecord
from tqdm import tqdm

from PhageScanner.main.exceptions import PipelineExceptionError
from PhageScanner.main.utils import CSVUtils, FastaUtils


def get_feature_names(dataframe: pd.DataFrame):
    """Get the feature names.

    Description:
        This method takes in a dataframe and
        returns the feature names. The expectation
        is that the default feature names
        are described by the default_column_names
        vector.
    """
    # get feature column names.
    default_column_names = [
        "accession",
        "length",
        "start_pos",
        "stop_pos",
        "orf_score",
        "protein",
    ]
    feature_column_names = ["ORF"]
    for col in dataframe.columns:
        if col not in default_column_names:
            feature_column_names.append(col)

    return feature_column_names


def get_feature_array(
    dataframe: pd.DataFrame, feature_column_names: List[str]
) -> List[Tuple[str, str, int, int]]:
    """Get a feature array.

    Description:
        This function returns a feature array that contains
        a list of tuples containing:
            1. Feature Type (str) - i.e. the column name
            2. Feature name (str)
            3. Start index (int)
            4. End index (int)
    """
    features_array = []
    for _, row in dataframe.iterrows():
        start_position = row["start_pos"]
        stop_position = row["stop_pos"]
        feature_tuple = (
            "ORF",
            f"ORF (start={start_position};stop={stop_position})",
            start_position,
            stop_position,
        )
        features_array.append(feature_tuple)
        for feature_type in feature_column_names:
            if feature_type == "ORF":
                continue
            feature_name = row[feature_type]
            features_array.append(
                (feature_type, feature_name, start_position, stop_position)
            )
    return features_array


def get_color_map(feature_array: List[str]):
    """Return a name to color map.

    Description:
        This function maps each feature to a shade
        of green.
    """
    shades_of_green = [
        "#1E5631",
        "#A4DE02",
        "#76BA1B",
        "#4C9A2A",
        "#ACDF87",
        "#ACDF87",
        "#68BB59",
        "#00FF7F",
        "#9ACD32",
        "#228B22",
    ]
    feature2color = {}
    for index, feature_type in enumerate(feature_array):
        index = index % len(shades_of_green)
        feature2color[feature_type] = shades_of_green[index]

    return feature2color


def create_genome_figure(
    features_array: List[Tuple[str, str, int, int]],
    output_figure_path: Path,
    sequence_len: int,
    feature2color: Dict[str, str],
):
    """Create a figure showing the features.

    Description:
        This method uses the library dna_features_viewer
        to create a figure that contains all of the
        features found from the pipeline.
    """
    feature_objects = []
    for feature_type, feature_name, start_pos, end_pos in features_array:
        feature = GraphicFeature(
            start=start_pos,
            end=end_pos,
            label=feature_name,
            strand=+1,
            color=feature2color[feature_type],
        )
        feature_objects.append(feature)

    record = GraphicRecord(sequence_length=sequence_len, features=feature_objects)

    ax, _ = record.plot(figure_width=100, figure_height=10)
    ax.figure.savefig(output_figure_path, bbox_inches="tight")

    # close figure to save memory
    plt.close(ax.figure)


def create_genome_images(
    path_to_predictions: Path, output_path: Path
):
    """Create images with features for each assembly/genome.

    Description:
        This function is the primary main function for creating
        images with features for each assembly/genome. This uses
        other functions for plotting features onto genomic coordinates.

    Args:
        path_to_predictions (Path): The path to the output predictions csv.
        output_path (Path): The path for storing the curated images.
    """
    # create output directory if it doesn't exist.
    if not output_path.exists() or len(os.listdir(output_path)) == 0:
        if not output_path.exists():
            os.mkdir(output_path)
    else:
        raise PipelineExceptionError(
            "Output directory must be empty and contain existing parent directories."
        )

    # turn the output csv into a dataframe.
    prediction_result_df = CSVUtils.csv_to_dataframe(path_to_predictions)

    # get list of unique contigs or genomes.
    accession_list = prediction_result_df["accession"].unique()

    # get prediction dataframe from csv file.
    for accession in tqdm(accession_list):
        # for name, sequence in FastaUtils.get_proteins(path_to_assemblies):
        #     if name == accession:
                # get the dataframe for the accession.
        accession_df = prediction_result_df[
            prediction_result_df["accession"] == accession
        ]

        # get the sequence length.
        sequence_length = accession_df['length'].max()

        # get feature column names.
        feature_column_names = get_feature_names(accession_df)
        # get colors for each feature column
        feature2color = get_color_map(feature_column_names)
        # extract features.
        features_array = get_feature_array(accession_df, feature_column_names)
        # create graphic objects from features.
        create_genome_figure(
            features_array=features_array,
            output_figure_path=output_path / f"{accession}.png",
            sequence_len=sequence_length,
            feature2color=feature2color,
        )
