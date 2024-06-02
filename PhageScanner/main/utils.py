""" Contains utility functions.

Description:
    This module contains utility functions that
    are used throughout the library for manipulating
    files, downloading serialized objects, uploading
    serialized objects, and other various methods.
"""

import csv
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import yaml

from PhageScanner.main.exceptions import (
    IncorrectValueError,
    IncorrectYamlError,
    PipelineCommandError,
)


def get_filename(filename: Union[str, Path]):
    """Return the filename.

    Description:
        returns the filename without the extension
        or filepath.
    """
    return os.path.splitext(os.path.basename(filename))[0]


class CommandLineUtils:
    """Class contains command-line utility functions."""

    @staticmethod
    def execute_command(command):
        """Execute command line.

        Description:
            This method executes a call to the command line.
            It assumes that the tools in the commmand are installed.

        Note:
            Here we use POpen to run the command in a shell. It's not
            expected there would be any security risk here, but it's generally
            not used in production environments.
        """
        # Use subprocess to execute the command
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )
        process.wait()

        # get the output of the command
        output, error = process.communicate()

        # NOTE: ignoring errors from phanotate since it throws errors without tscan.
        # NOTE: ignoring errors from megahit since it throws errors even on success.
        if (
            len(error) > 0
            and not command.startswith("phanotate.py")
            and not command.startswith("megahit")
        ):
            error_message = "There was an error executing a shell command.\n"
            error_message += f"The error was: \n\n{error}\n\n"
            error_message += f"The command was: \n\n{command}\n\n"
            raise PipelineCommandError(error_message)

        return output


class LogUtils(object):
    """Wrapper around the logging library."""

    @staticmethod
    def configure_logging(path_to_log: Path, log_level: str):
        """Instantiate a logger."""
        # create directory if it doesn't exist
        if not os.path.exists(path_to_log.parent):
            os.mkdir(path_to_log.parent)

        # set verbose logging level and output format
        verbosity2loglevel = {"info": logging.INFO, "debug": logging.DEBUG}
        logging_format = (
            "%(levelname)s - %(name)s:%(filename)s:%(lineno)d - %(message)s"
        )

        # set up logging for file.
        logging.basicConfig(
            filename=path_to_log,
            filemode="w",
            format=logging_format,
            level=verbosity2loglevel[log_level],
        )

        # set up logging to console
        console = logging.StreamHandler()
        console.setLevel(verbosity2loglevel[log_level])
        formatter = logging.Formatter(logging_format)
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)

        # add results level to logger
        logging.addLevelName(25, "RESULT")

        logging.debug(f"Logger has been configured: {path_to_log}")


class ConfigUtils:
    """Methods for configuration files."""

    @staticmethod
    def open_config(config_file: Path):
        """Open a configuration file.

        Description:
            opens a configuration file specified in the
            yaml format.
        """
        try:
            with open(config_file, "r") as file:
                data = yaml.safe_load(file)
        except Exception as ex:
            raise IncorrectYamlError(
                f"There is an error opening the config! Error: \n\n {ex}"
            )
        return data


class DatabaseConfig(ConfigUtils):
    """The database configuration object.

    Description:
        This class defines the database configuration
        object. To change how the pipeline works with the
        configuration, this must be updated.
    """

    def __init__(self, config_file: Path):
        """Open a configuration file.

        Description:
            opens a configuration file specified in the
            yaml format.
        """
        self.config = super().open_config(config_file)

    def get_classes(self):
        """Get the sequence classes for database construction."""
        return self.config.get("classes")

    def get_clustering_tool(self):
        """Get the clustering tool name."""
        return self.config["clustering"]["name"]

    def get_clustering_threshold(self):
        """Get the clustering identity threshold"""
        return self.config["clustering"]["clustering-percentage"] / 100

    def get_k_partition_count(self) -> int:
        """Get the number of partitions to use for k-fold CV."""
        return self.config["clustering"]["k_partitions"]

    def get_deduplication_threshold(self) -> int:
        """Get the number of partitions to use for k-fold CV."""
        return self.config["clustering"]["deduplication-threshold"] / 100


class TrainingConfig(ConfigUtils):
    """The training pipeline configuration object.

    Description:
        This class defines the training configuration
        object. To change how the pipeline works with the
        configuration, this must be updated.
    """

    def __init__(self, config_file: Path):
        """Open a configuration file.

        Description:
            opens a configuration file specified in the
            yaml format.
        """
        self.config = super().open_config(config_file)

    def get_classes(self):
        """Get the sequence classes for training."""
        classes = []
        for class_info in self.config.get("classes"):
            class_name = class_info.get("name")
            csv_path = Path(class_info.get("final_csv"))
            classes.append((class_name, csv_path))
        return classes

    def get_models(self):
        """Get the models for training.

        Description:
            returns the model names for the training.
        """
        models = [m["name"] for m in self.config["models"]]
        return models

    def get_model_features(self, model_name):
        """Get the model features for a given model."""
        for m in self.config["models"]:
            if m["name"] == model_name:
                for feature_info in m["features"]:
                    feature_name = feature_info["name"]
                    if "parameters" in feature_info:
                        parameters = feature_info["parameters"]
                    else:
                        parameters = None
                    yield feature_name, parameters

    def sequential(self, model_name):
        """Return True if the model takes in sequential data."""
        for m in self.config["models"]:
            if m["name"] == model_name:
                if m["model_info"]["sequential"]:
                    return m["model_info"]["sequential"]
        return False

    def get_predictor_model_name(self, model_name):
        """Get the name of the predictor from the model name."""
        model_predictor_name = None
        for m in self.config["models"]:
            if m["name"] == model_name:
                model_predictor_name = m["model_info"]["model_name"]
                break

        if model_predictor_name is None:
            raise IncorrectYamlError(
                f"A model with the name {model_name} does not exist."
            )

        return model_predictor_name


class PredictionConfig(ConfigUtils):
    """The prediction pipeline configuration object.

    Description:
        This class defines the prediction pipeline
        configuration file and methods. This class
        is directly tied to the yaml file for
        configuration of the prediction pipeline.
    """

    def __init__(self, config_file: Path):
        """Open a configuration file.

        Description:
            opens a configuration file specified in the
            yaml format.
        """
        self.config = super().open_config(config_file)

    def get_model_names(self):
        """Get a list of model names."""
        models = [m["name"] for m in self.config["models"]]
        return models

    def get_assembler_name(self):
        """Get the name of the assembler from the configuration."""
        return self.config["assembler"]

    def get_orffinder_name(self):
        """Get the name of the orffinder from the configuration."""
        return self.config["orffinder"]

    def get_probability_threshold(self):
        """Get the probability needed for classifying proteins."""
        return float(self.config["probability_threshold"])

    def sequential(self, model_name):
        """Return True if the model takes in sequential data."""
        for m in self.config["models"]:
            if m["name"] == model_name:
                if m["model_info"]["sequential"]:
                    return m["model_info"]["sequential"]
        return False

    def get_model_path(self, model_name):
        """Get the model path from the model name."""
        for m in self.config["models"]:
            if m["name"] == model_name:
                return m["model_path"]
        # if nothing found
        raise IncorrectYamlError(f"A model with the name {model_name} does not exist.")

    def is_sequential(self, model_name):
        """Return True if the model takes in sequential data."""
        for m in self.config["models"]:
            if m["name"] == model_name:
                if m["sequential"]:
                    return True
        return False

    def get_model_classes(self, model_name):
        """Get the model path from the model name."""
        for m in self.config["models"]:
            if m["name"] == model_name:
                return m["index2class_file"]
        # if nothing found
        raise IncorrectYamlError(f"A model with the name {model_name} does not exist.")

    def get_model_features(self, model_name):
        """Get the model features for a given model."""
        for m in self.config["models"]:
            if m["name"] == model_name:
                for feature_info in m["features"]:
                    feature_name = feature_info["name"]
                    if "parameters" in feature_info:
                        parameters = feature_info["parameters"]
                    else:
                        parameters = None
                    yield feature_name, parameters

    def get_predictor_model_name(self, model_name):
        """Get the name of the predictor from the model name."""
        model_predictor_name = None
        for m in self.config["models"]:
            if m["name"] == model_name:
                model_predictor_name = m["model_info"]["model_name"]
                break

        if model_predictor_name is None:
            raise IncorrectYamlError(
                f"A model with the name {model_name} does not exist."
            )

        return model_predictor_name


class CSVUtils:
    """Methods for CSV files."""

    @staticmethod
    def count_rows(csv_file: Path):
        """Return the number of rows in a csv file."""
        lines = 0
        with open(csv_file, "r") as f:
            while f.readline() is not None:
                lines += 1
        return lines

    @staticmethod
    def csv_to_dataframe(csv_file: Path):
        """Convert a CSV file to a dataframe."""
        return pd.read_csv(csv_file, sep=",")

    @staticmethod
    def appendcsv(
        data_dict: List[Dict[str, Union[str, float, int]]],
        fieldnames: List[str],
        file_path: Path,
    ):
        """Convert a string formatted as CSV to fasta formatted output."""
        # Data validation
        if not isinstance(data_dict, list):
            raise IncorrectValueError("Input data must be a list of dictionaries.")
        for d in data_dict:
            if not isinstance(d, dict):
                raise IncorrectValueError(
                    "Each element in the list must be a dictionary."
                )

        # check if file exists and isn't empty
        file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0

        # if header already exists in the file, use that.
        if file_exists:
            with open(file_path, "r") as file:
                fieldnames = file.readline().strip("\n").split(",")

        # create output csv file.
        with open(file_path, "a", newline="\n") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # write header.
            if not file_exists:
                try:
                    writer.writeheader()
                except Exception as e:
                    err_msg = "Error writing header to CSV! "
                    err_msg += "Check that the file was created. "
                    err_msg += f"Full error: \n\n {e}"
                    raise IncorrectValueError(err_msg)

            # write each row.
            for row in data_dict:
                try:
                    writer.writerow(row)
                except Exception as e:
                    err_msg = "Error writing row to CSV! "
                    err_msg += "Check that the columns match, and the file is saved. "
                    err_msg += f"Full error: \n\n {e}"
                    raise IncorrectValueError(err_msg)


class FastaUtils:
    """Methods for Fasta files."""

    @staticmethod
    def fasta_to_dataframe(fasta_file: Path):
        """Convert a fasta file to a dataframe."""
        protein_array = []
        for accession, protein in FastaUtils.get_proteins(fasta_file):
            protein_array.append([accession, protein])
        return pd.DataFrame(protein_array, columns=["accession", "protein"])

    @staticmethod
    def count_entries_in_fasta(fasta_file):
        """Get the number of entries in a fasta file."""
        with open(fasta_file, "r") as file:
            fasta_contents = file.read()
            return fasta_contents.count(">")

    @staticmethod
    def get_proteins(fasta_path: Path, withfullname=False):
        """Get accesion and protein from fasta file.

        Description:
            This method returns an accession and protein
            one at a time using a generator. This ensures
            the entire fasta file is not stored in memory
            at once.

        Returns/Yields:
            accession (string)
            protein (string)
        """
        with fasta_path.open() as file:
            accession, protein = None, []
            for line in file:
                line = line.rstrip()
                if line.startswith(">"):
                    if accession:
                        yield accession, "".join(protein)
                    if withfullname:
                        accession, protein = line[1:], []
                    else:
                        accession, protein = line[1:].split(" ")[0], []
                else:
                    protein.append(line)
            if accession:
                yield accession, "".join(protein)
