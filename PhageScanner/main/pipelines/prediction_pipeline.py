"""The prediction pipeline uses trained models to annotate genomes."""

import csv
import logging
import os
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from PhageScanner.main import utils
from PhageScanner.main.DNA import DNA
from PhageScanner.main.feature_extractors import ProteinFeatureExtraction
from PhageScanner.main.models import ModelNames
from PhageScanner.main.pipelines.pipeline_interface import Pipeline
from PhageScanner.main.tool_wrappers.assembler_wrappers import AssemblyWrapperNames
from PhageScanner.main.tool_wrappers.orffinder_wrappers import OrfFinderWrapperNames
from PhageScanner.main.utils import CSVUtils, FastaUtils


class PredictionPipelineType(Enum):
    """Enum representing the prediction pipeline types.

    Description:
        Defines the different possible prediction pipeline types.
        available. Examples of possible prediction pipelines
        are reads, proteins and genomes.
    """

    reads = "reads"
    genome = "genome"
    proteins = "protein"

    @staticmethod
    def get_type(name: str):
        """Get the type of the prediction pipeline."""
        # Iterate over all members of the enum
        for pipeline_type in PredictionPipelineType:
            if name == pipeline_type.value:
                logging.info(f"prediction type: {pipeline_type}")
                return pipeline_type
        raise ValueError("Unknown prediction pipeline type")


class PredictionPipeline(Pipeline):
    """A pipeline for predicting protein classes.

    Description:
        This class controls the prediction pipeline. This
        pipeline takes in a set of models for performing
        predictions for different types of protein classes.
        This allows for input to be me unassembled reads,
        fully assembled genomes, or fasta files of proteins.
    """

    def __init__(
        self,
        input: Path,
        input_type: str,
        config: Path,
        orf_finder_tool_path: Path,
        assembler_tool_path: Path,
        training_output_directory: Path,
        probability_threshold: float,
        pipeline_name: str,
        directory: Path,
    ):
        """Initialize the training pipeline."""
        logging.info("Running PredictionPipeline | Creating pipeline...")
        self.config_object = utils.PredictionConfig(config)
        self.pipeline_name = pipeline_name
        self.directory = directory
        self.input = input
        self.input_type = PredictionPipelineType.get_type(input_type)
        self.probability_threshold = probability_threshold
        self.training_output_directory = training_output_directory

        # create the ORF-finding and assembler tool wrappers
        self.orffinder = OrfFinderWrapperNames.get_orffinding_tool(orf_finder_tool_path)
        self.assembler = AssemblyWrapperNames.get_assembly_tool(assembler_tool_path)

        # dataframe for performing predictions.
        self.dataframe = None

        # create directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def assemble_reads(self):
        """Assemble the reads.

        Description:
            Assemble the reads input into the pipeline. These
            reads are assembles so later steps in the pipeline
            can retrieve ORFs from the assemblies.
        """
        logging.info(f"Assembling reads: {self.input}")

        # use the assembler to assemble the reads.
        assembly_file_prefix = str(self.pipeline_name) + "_" + "assembly"
        assembly_path = self.directory / assembly_file_prefix
        full_assembly_path = self.assembler.assemble(
            first=self.input, out_directory=assembly_path
        )

        return full_assembly_path

    def get_proteins(self):
        """Get the proteins from an input set of contigs/genomes.

        Description:
            This method uses a path to the either contigs or genomes
            and retrieves ORFs per. These ORFs are the converted into
            the expected protein products.
        """
        # eventual outputs
        orf_path = self.directory / (str(self.pipeline_name) + "_" + "orfs")
        orf_file_prefix_proteins = self.directory / (
            str(self.pipeline_name) + "_" + "orfs" + "_proteins.csv"
        )

        # check if completed..
        if (
            os.path.exists(orf_file_prefix_proteins)
            and os.stat(orf_file_prefix_proteins).st_size > 0
        ):
            message = f"(Skipping) Found ORF file CSV: '{orf_file_prefix_proteins}'. "
            message += "Please delete this file if you'd like to rerun!"
            logging.info(message)
            return orf_file_prefix_proteins

        # Get ORFs.
        orf_file_path = self.orffinder.find_orfs(fasta_path=self.input, outpath=orf_path)

        # get length of each contig/genome
        accession2length = {}
        for contig_name, sequence in FastaUtils.get_proteins(self.input):
            accession2length[contig_name] = len(sequence)

        # turn ORFs into proteins.
        logging.info(f"saving ORF-converted proteins to: {orf_file_prefix_proteins}")
        for orf_name, orf_sequence in FastaUtils.get_proteins(
            orf_file_path, withfullname=True
        ):
            accession_id, start_pos, stop_pos, score = self.orffinder.get_info_from_name(
                orf_name
            )

            # save to output file.
            orf_results = {
                "accession": accession_id,
                "length": accession2length[accession_id],
                "start_pos": start_pos,
                "stop_pos": stop_pos,
                "orf_score": score,
                "protein": DNA.dna2protein(orf_sequence),
            }
            CSVUtils.appendcsv(
                data_dict=[orf_results],  # input must be an array.
                fieldnames=orf_results.keys(),
                file_path=orf_file_prefix_proteins,
            )

        return orf_file_prefix_proteins

    def get_predictions(self):
        """Get predictions for protein sequences.

        Description:
            This method returns predictions for input proteins.
            The `self.dataframe` MUST be a pandas dataframe containing
            a `protein` column. That's it. The output from these
            predictions are saved to the specified output directory.
        """
        # clean the proteins
        logging.info("Cleaning proteins...")
        self.dataframe["protein"] = self.dataframe["protein"].apply(
            ProteinFeatureExtraction.clean_protein
        )
        logging.info("Cleaning proteins...")

        # for each model, get predictions.
        for model in self.config_object.get_model_names():
            # extract features from each protein.
            self.extract_feature_vector(model)

            # instantiate the model.
            model_path = self.config_object.get_model_path(model, self.training_output_directory)
            model_predictor_name = self.config_object.get_predictor_model_name(model)
            model_object = ModelNames.get_model(model_predictor_name)
            model_object = model_object.load(model_path)

            # get the model classes.
            index2class = self.config_object.get_model_classes(model)

            # predict features from proteins.
            x_test = np.vstack(self.dataframe["features"].to_numpy())
            predictions, probabilities = model_object.predict(x_test)
            self.dataframe[model] = predictions

            # use probabilities threshold.
            if len(probabilities) == len(predictions):
                probabilities_series = pd.Series(probabilities)
                self.dataframe[model] = self.dataframe[model].mask(
                    probabilities_series < self.probability_threshold,
                    -1,
                )

            # replace values with class names.
            self.dataframe[model] = self.dataframe[model].replace(index2class)

    def run(self):
        """Run the prediction pipeline."""
        # Step 1: if reads, assemble the reads.
        if self.input_type == PredictionPipelineType.reads:
            logging.info("Step 1 - Assemble the input reads...")
            self.input = self.assemble_reads()  # updates input (as if genomes)
            logging.info("Step 1 (Finished) - Assemble the input reads..")

        # if reads or contigs/genomes, find ORFs and convert to protein.
        iscontigs = (
            self.input_type == PredictionPipelineType.genome
            or self.input_type == PredictionPipelineType.reads
        )
        if iscontigs:
            # filter for bacteriophage contigs. TODO: implement this.
            # find the open reading frames.
            logging.info("Step 3 - Get proteins from predicted ORFs...")
            self.input = self.get_proteins()  # updates input (as if genomes)
            logging.info("Step 3 (Finished) - Get proteins from predicted ORFs...")

        # get the protein predictions using models.
        if iscontigs:
            self.dataframe = pd.read_csv(self.input)
        else:
            self.dataframe = FastaUtils.fasta_to_dataframe(self.input)

        # make predictions
        self.get_predictions()

        # save the predictions.
        final_output_path = self.directory / (
            str(self.pipeline_name) + "_" + "predictions.csv"
        )
        self.dataframe = self.dataframe.drop(["features"], axis=1, errors="ignore")
        self.dataframe.to_csv(final_output_path, sep=",", index=False)
