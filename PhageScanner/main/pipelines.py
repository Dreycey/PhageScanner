""" Contains pipelines used by the ML pipeline.

Description:
    This module contains different pipelines
    and an interface for creating new pipelines.

TODO:
    1. Create a string enum for calling columns in dataframe
    2. Allow for feature selection
"""
import csv
import logging
import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from PhageScanner.main import database_adapters, utils
from PhageScanner.main.assembler_wrappers import AssemblyWrapperNames
from PhageScanner.main.clustering_wrappers import ClusteringWrapperNames
from PhageScanner.main.database_adapters import DatabaseAdapterNames
from PhageScanner.main.DNA import DNA
from PhageScanner.main.exceptions import IncorrectValueError
from PhageScanner.main.feature_extractors import (
    FeatureExtractorNames,
    ProteinFeatureAggregator,
    ProteinFeatureExtraction,
    SequentialProteinFeatureAggregator,
)
from PhageScanner.main.models import ModelNames
from PhageScanner.main.orffinder_wrappers import OrfFinderWrapperNames
from PhageScanner.main.utils import CSVUtils, FastaUtils


class PipelineNames(Enum):
    """Names of pipeline adapters.

    Description:
        This enum contains the names of each pipeline.
    """

    database = "database"
    train = "train"
    predict = "predict"


class Pipeline(ABC):
    """Abstract class for creating new pipelines"""

    # get time of the pipeline run
    pipeline_start_time = time.ctime()

    @abstractmethod
    def run(self):
        """Run the pipeline."""
        pass

    def extract_feature_vector(self, model_name):
        """Extract the feature vector from each protein.

        Description:
            This method extracts feature vectors for each protein in
            the dataframe "self.dataframe".
        """
        logging.info(f"extracting protein features: '{model_name}' d")

        # get feature extractors.
        feature_list = []
        for feature_name, parameters in self.config_object.get_model_features(
            model_name
        ):
            extractor = FeatureExtractorNames.get_extractor(feature_name, parameters)
            feature_list.append(extractor)

        # create feature aggregator (combines features)
        segment_size = self.config_object.sequential(model_name)
        if segment_size:
            aggregator = SequentialProteinFeatureAggregator(
                extractors=feature_list, segment_size=segment_size
            )
        else:
            aggregator = ProteinFeatureAggregator(extractors=feature_list)

        # use the aggregator to extract features from self.dataframe
        logging.info(f"extracting features for model: '{model_name}': {feature_list}")
        logging.info(f"extractor's features: '{aggregator.extractors}")
        self.dataframe["features"] = self.dataframe["protein"].apply(
            aggregator.extract_features
        )

        logging.info(f"done extracting features for model: '{model_name}'")


class DatabasePipeline(Pipeline):
    """An database creation pipeline.

    Description:
        This class controls the creation, curation,
        and preprocessing steps for the protein retrieval.
    """

    def __init__(self, config: Path, pipeline_name: str, directory: Path):
        """Initialize the database pipeline."""
        logging.info("Running DatabasePipeline | Creating pipeline...")
        self.config_object = utils.DatabaseConfig(config)
        self.pipeline_name = pipeline_name
        self.directory = directory

        # The following extension is used for clustered proteins.
        self.cluster_extension = "_clustered"

        # create directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def get_fasta_path(self, class_name):
        """Get the fasta path for proteins before clustering."""
        path = self.directory / (self.pipeline_name + "_" + class_name + ".fasta")
        return path

    def get_clustered_fasta_path(self, class_name):
        """Get the fasta path for proteins after clustering."""
        full = self.pipeline_name + "_" + class_name + self.cluster_extension
        path = self.directory / full
        return path

    def get_partition_csv_path(self, class_name):
        """Get the fasta path for proteins after clustering."""
        full = self.pipeline_name + "_" + class_name + ".csv"
        path = self.directory / full
        return path

    def get_proteins_from_db_adapters(self):
        """Get proteins from each database adapter and saves to local file.

        Description:
            This method is used to retrieved the proteins from each database.
            It works by going through each class, then iterating through each
            specified database query. The sequences are then appended to a fasta
            file.

        Note:
            This will not overwrite existing files with the name. Instead, it assumes
            that the existing file has already been created. This allows for rerunning
            the pipeline without having to start from scratch.
        """
        logging.info("Using database adapters to retrieve proteins.")

        for class_info in self.config_object.get_classes():
            class_name = class_info.get("name")
            logging.info(f"\t Retieving proteins for class: {class_name}")

            # create a file for storing the proteins.
            full_path = self.get_fasta_path(class_name)

            # if the file already exists, then go to next class.
            if os.path.isfile(full_path):
                logging.warning(
                    f"(Skip) Class already obtained: {class_name} | {full_path}"
                )
                continue

            # perform a query against each specified database adapter.
            with open(full_path, "a") as protein_class_file:
                for database_name, query in class_info.items():
                    if database_name == "name":
                        continue
                    logging.info(
                        f"\t Getting {class_name} proteins using DB: {database_name}"
                    )

                    # Get adapter with the name in the configuration file.
                    db_adapter = DatabaseAdapterNames.get_db_adapter(database_name)
                    if database_name == DatabaseAdapterNames.entrez.value:
                        query = database_adapters.EntrezAdapter.get_phanns_query(query)

                    # query the database adapter
                    logging.debug(f"\t\t Database: {database_name}, query: {query}")
                    count = 0
                    for batch in db_adapter.query(query=query):
                        if len(batch) > 0:
                            protein_class_file.write(batch)
                            count += batch.count(">")
                        else:
                            logging.warning(
                                f"Empt Batch! DB: {database_name}, query: {query}"
                            )

                    # report the count
                    logging.log(
                        logging.getLevelName("RESULT"),
                        f"COUNT DB:{database_name}, Class:{class_name}, Count:{count}",
                    )

                    # save count to csv.
                    db_count_csv = self.directory / "db_count.csv"
                    temp_db_count = {
                        "datetime": self.pipeline_start_time,
                        "database": database_name,
                        "class": class_name,
                        "class_count": count,
                    }
                    CSVUtils.appendcsv(
                        data_dict=[temp_db_count],  # input must be an array.
                        fieldnames=temp_db_count.keys(),
                        file_path=db_count_csv,
                    )

    def cluster_proteins(self):
        """Cluster each class of proteins.

        Description:
            This method clusters all proteins pertaining to each
            class.
        """
        clustering_tool = self.config_object.get_clustering_tool()
        clustering_adapter = ClusteringWrapperNames.get_clustering_tool(clustering_tool)
        # for each class fasta file, cluster the proteins.
        for filename in os.listdir(self.directory):
            if self.cluster_extension in filename or ".fasta" not in filename:
                continue
            class_fasta_file = os.path.join(self.directory, filename)
            class_clstr_file = os.path.join(
                self.directory, utils.get_filename(filename) + self.cluster_extension
            )

            # if the file already exists, then go to next class.
            if os.path.isfile(class_clstr_file):
                logging.warning(
                    f"(Skip) These have already been clustered: {class_clstr_file}"
                )
                continue
            else:
                logging.info(f"Clustering proteins in {filename}")

            # cluster proteins.
            clustering_adapter.cluster(
                fasta_file=class_fasta_file,
                outpath=class_clstr_file,
                identity=self.config_object.get_clustering_threshold(),
            )

            # save count to csv.
            db_count_csv = self.directory / "result_cluster_ouput.csv"
            temp_db_count = {
                "datetime": self.pipeline_start_time,
                "class_name": filename.split("_")[1].replace(".fasta", ""),
                "clustering_threshold": self.config_object.get_clustering_threshold(),
                "cluster_count": FastaUtils.count_entries_in_fasta(
                    fasta_file=class_clstr_file
                ),
            }
            CSVUtils.appendcsv(
                data_dict=[temp_db_count],  # input must be an array.
                fieldnames=temp_db_count.keys(),
                file_path=db_count_csv,
            )

    def partition_proteins(self, k_partitions=5, get_cluster_sizes=False):
        """Partion the proteins.

        Description:
            Partitions the proteins allowing for the downstream training/testing
            pipeline to utilize k-fold cross validation during testing. This is
            done by first splitting the clustered proteins into k partitions for
            each specified class. Thereafter the the cluster members for each class
            added to each partition. The end result is a CSV file that paritions
            that have little similarity between partitions, regardless of the class.

        Returns:
            A CSV per class.
            columns:
                1. partition number
                2. protein length
                3. protein name
                4. sequence
        """
        for class_info in self.config_object.get_classes():
            class_name = class_info.get("name")  # TODO: move to config_object
            logging.info(f"\t Partitioning class: {class_name}")

            # get path to proteins before and after clustering.
            fasta_non_clustered = self.get_fasta_path(class_name)
            fasta_clustered = self.get_clustered_fasta_path(class_name)

            # get clustering tool
            clustering_tool = self.config_object.get_clustering_tool()
            clustering_adapter = ClusteringWrapperNames.get_clustering_tool(
                clustering_tool
            )

            # get clusters as Dict
            # TODO: should done without storing all clusters into  memory.
            cluster_graph = clustering_adapter.get_clusters(fasta_clustered)

            # randomize the clusters
            randomized_clusters = list(cluster_graph.keys())
            np.random.shuffle(randomized_clusters)

            # save cluster sizes to csv
            if get_cluster_sizes:
                cluster_count_csv = self.directory / "cluster_sizes.csv"
                temp_cluster_count = {
                    "datetime": self.pipeline_start_time,
                    "class_name": class_name,
                    "cluster_count": len(cluster_graph.keys()),
                    "cluster_sizes": '\t'.join([str(len(cluster)) for cluster in cluster_graph.values()])
                }
                CSVUtils.appendcsv(
                    data_dict=[temp_cluster_count],
                    fieldnames=temp_cluster_count.keys(),
                    file_path=cluster_count_csv,
                )

            # obtain a dictionary of protein -> partition
            protein2partition = {}
            for i, cluster_id in enumerate(randomized_clusters):
                cluster_partition = (i % k_partitions) + 1
                # assign clusters to the same partition
                for protein_accesion in cluster_graph[cluster_id]:
                    protein2partition[protein_accesion] = cluster_partition

            # delete graph to save some space
            del randomized_clusters
            del cluster_graph

            # get accesion ids for each clusters reference/centroid protein.
            output_file = self.get_partition_csv_path(class_name)
            with open(output_file, "w") as output_csv:
                output_csv.write("partition,accession,protein,protein_length\n")
                for accession, protein in FastaUtils.get_proteins(fasta_non_clustered):
                    if accession[:19] in protein2partition:
                        partition = protein2partition[accession[:19]]
                        output_csv.write(
                            f"{partition},{accession},{protein},{len(protein)}\n"
                        )
                    else:
                        logging.warning(
                            f"protein {accession} was not found in clusters"
                        )

    def run(self):
        """Run the pipeline.

        Description:
            This method runs the pipeline for
            creating a new set of unique proteins
            using different input database adapters.
        """
        # Step 1: get proteins using database adapters.
        logging.info("Step 1 - Obtaining proteins from APIs...")
        self.get_proteins_from_db_adapters()
        logging.info("Step 1 (Finished) - Obtaining proteins from APIs...")

        # Step 2: cluster proteins at the predifined clustering threshold.
        logging.info("Step 2 - Cluster the proteins...")
        self.cluster_proteins()
        logging.info("Step 2 (Finished) - Cluster the proteins...")

        # Step 3: create k-fold partitioned clusters
        logging.info("Step 3 - Create k-fold partitions...")
        self.partition_proteins()
        logging.info("Step 3 (Finished) - Create k-fold partitions...")


class TrainingPipeline(Pipeline):
    """A pipeline for training and testing ML models.

    Description:
        This class controls the training and testing
        of machine learning models specified in the
        input config.
    """

    def __init__(self, config: Path, pipeline_name: str, directory: Path):
        """Initialize the training pipeline."""
        logging.info("Running TrainingPipeline | Creating pipeline...")
        self.config_object = utils.TrainingConfig(config)
        self.pipeline_name = pipeline_name
        self.directory = directory
        self.number_of_classes = 0  # updated during pipeline run.

        # pandas dataframe holds all of the training data
        self.dataframe = pd.DataFrame()

        # dictionary to map index to class name
        self.class2name = dict()

        # create directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def combine_partitions(self):
        """Combine the partitions for different classes CSVs.

        Description:
            This method combines classes into different partitions.
            The output of using this method void, since the self.dataframe
            is added to for each input. In addition, the input value ("value")
            is added to each input dataframe, for the new column "class",
            before adding to self.dataframe.

        Args:
            data (.csv file path): contains a csv file path.
        """
        index = 0
        # get classes
        model_classes = self.config_object.get_classes()

        # save number of classes
        self.number_of_classes = len(model_classes)
        for class_name, class_path in model_classes:
            # Read the csv file into a pandas DataFrame
            df = pd.read_csv(class_path)

            # Add a new column 'class' with the provided value
            # NOTE: index must be an integer.
            df["class"] = index

            # add information about class to name mapping
            # NOTE: index must be an integer.
            self.class2name[index] = class_name

            # Append this data to the main dataframe
            # NOTE: this is not optimal since Pandas
            # DFs will result in quadratic behaviour due
            # to duplicating the dataframe. However,
            # here we expect few classes.
            # see: https://stackoverflow.com/questions/36489576/
            self.dataframe = pd.concat([self.dataframe, df])

            index += 1

        # only select columns needed
        try:
            self.dataframe = self.dataframe[["partition", "class", "protein"]]
        except KeyError as ex:
            raise IncorrectValueError(
                f"Columns in {class_path} are not correct: {ex.message}"
            )

    def balance_partitions(self):
        """Balance classes  within each partition."""
        new_balanced_partitions_df = []
        for partition in self.dataframe["partition"].unique():
            partition_df = self.dataframe[self.dataframe["partition"] == partition]

            # get smallest class size.
            class_sizes = partition_df["class"].value_counts()
            min_class = min(class_sizes)

            # create a balanced partition.
            balanced_partition_df_list = []
            for class_index in partition_df["class"].unique():
                class_df = partition_df[partition_df["class"] == class_index]
                balanced_class_df = class_df.sample(n=min_class, random_state=1)
                balanced_partition_df_list.append(balanced_class_df)

            # add balanced partition to new_balanced_partitions_df
            balanced_partition_df = pd.concat(
                balanced_partition_df_list, ignore_index=True
            )
            assert min_class == min(
                balanced_partition_df["class"].value_counts()
            )  # TODO: turn into raise error
            new_balanced_partitions_df.append(balanced_partition_df)

        # update the entire dataframe
        self.dataframe = pd.concat(new_balanced_partitions_df, ignore_index=True)

    def get_kfold_training(self):
        """Get X_train, Y_train, X_test, Y_test.

        Description:
            Obtain training data split into sets that allow
            for training and testing to be performed based
            on how the data is partitioned.
        """
        unique_partitions = self.dataframe["partition"].unique()
        for test_partition in unique_partitions:
            logging.debug("Obtaining training splits.")
            # get partition.
            test = self.dataframe[self.dataframe["partition"] == test_partition]
            train = self.dataframe[self.dataframe["partition"] != test_partition]

            # get testing and training splits.
            x_test = np.vstack(test["features"].to_numpy())
            y_test = test["class"].to_numpy()
            x_train = np.vstack(train["features"].to_numpy())
            y_train = train["class"].to_numpy()

            logging.debug("Done: Obtaining training splits.")
            yield x_train, y_train, x_test, y_test

    def save_class_mapping(self):
        """Save class mapping."""
        # obtain information to save about class mapping.
        index2class_name_file = self.directory / "index2class_name.csv"
        output_index2class = {"datetime": self.pipeline_start_time}

        # convert keys to string for saving.
        for index, class_name in self.class2name.items():
            output_index2class[str(index)] = class_name

        # save the index to class mapping.
        CSVUtils.appendcsv(
            data_dict=[output_index2class],  # input must be an array.
            fieldnames=output_index2class.keys(),
            file_path=index2class_name_file,
        )

    def save_model_results(
        self,
        model_name: str,
        iteration: int,
        model_results: Dict[str, Union[str, float, str]],
    ):
        """Save the model results.

        Description:
            For all model results, save these to specific files.
            Some model results include:
                1. Confusion Matrix.
                2. Model F1 score, precision, recall, accuracy.
                    - These are saved for each k-fold tested.
        """
        # save the confusion matrix
        confusion_matrix_output_path = (
            self.directory / f"{model_name}_confusion_matrix.csv"
        )
        np.savetxt(
            confusion_matrix_output_path,
            model_results["confusion_matrix"],
            delimiter=",",
            fmt="%.3f",
        )
        del model_results["confusion_matrix"]

        # Save the f1score, accuracy, and precision
        output_path = self.directory / "model_results.csv"
        csv_file_headers = [
            "datetime",
            "model",
            "kfold_iteration",
            "accuracy",
            "f1score",
            "precision",
            "recall",
            "execution_time_seconds",
            "features",
            "dataset_size",
        ]

        # obtain a list of features and parameters.
        features = []
        for feature, params in self.config_object.get_model_features(model_name):
            if params:
                param_strs = [f"{k}={v}" for k, v in params.items()]
                params_text = "; ".join(param_strs)
                feature_string = f"{feature} ({params_text})"
            else:
                feature_string = feature
            features.append(feature_string)

        # add columns not originally present in the dataframe
        model_results["datetime"] = self.pipeline_start_time
        model_results["model"] = model_name
        model_results["kfold_iteration"] = iteration
        model_results["features"] = features

        # convert columns with many entries to strings
        for key in ["f1score", "precision", "recall", "features"]:
            model_results[key] = "\t".join(map(str, model_results[key]))

        # save results to disk.
        CSVUtils.appendcsv(
            data_dict=[model_results],  # input must be an array.
            fieldnames=csv_file_headers,
            file_path=output_path,
        )

    def run(self):
        """Run the training pipeline."""
        # Step 1: combine partitions from different classes.
        logging.info("Step 1 - Class balance and combine partitions...")
        self.combine_partitions()
        self.save_class_mapping()
        logging.info("Step 1 (Finished) - Class balance and combine partitions...")

        # Step 2: Balance the classes with different partitions.
        logging.info("Step 2 - Balancing classes in each partition...")
        self.balance_partitions()
        logging.info("Step 2 (Finished) - Balancing classes in each partition...")

        # Step 3: clean proteins
        logging.info("Step 3 - Cleaning proteins...")
        self.dataframe["protein"] = self.dataframe["protein"].apply(
            ProteinFeatureExtraction.clean_protein
        )
        logging.info("Step 3 (Finished) - Cleaning proteins...")

        # Step 3: extract features and train.
        logging.info("Step 4 - Training Models")
        models = self.config_object.get_models()
        for iteration, model_name in enumerate(models):
            # obtain features.
            logging.info(f"Step 4.{iteration} - Feature Extraction...")
            self.extract_feature_vector(model_name)
            logging.info(f"Step 4.{iteration} (Finished) - Feature Extraction...")

            # TODO: feature selection

            # train model
            logging.info(f"Step 4.{iteration} - k-fold testing model: {model_name}...")
            kfold_iteration = 0
            for x_train, y_train, x_test, y_test in self.get_kfold_training():
                # obtain model.
                model_predictor_name = self.config_object.get_predictor_model_name(
                    model_name
                )
                model_object = ModelNames.get_model(model_predictor_name)

                # train model.
                logging.info(f"Training Model: {model_name}...")
                model_object.train(x_train, y_train)
                logging.info(f"(Finished) Training Model: {model_name}...")

                # test trained model.
                logging.info(f"Testing Model: {model_name}...")
                model_results = model_object.test(x_test, y_test)
                logging.info(f"(Finished) Testing Model: {model_name}...")

                # save model results.
                self.save_model_results(
                    model_name=model_name,
                    iteration=kfold_iteration,
                    model_results=model_results,
                )

                # save the model.
                path2savemodel = self.directory / f"{model_name}"
                model_object.save(path2savemodel)

                # make sure saved model works.
                new_model = model_object.load(path2savemodel)
                new_model.test(x_test, y_test)

                # increment the kfold iteration
                kfold_iteration += 1

            logging.info(
                f"Step 4.{iteration} (Finished) - k-fold testing model: {model_name}..."
            )


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
                logging.info("prediction type: {pipeline_type}")
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

        # get assembler
        assembler_name = self.config_object.get_assembler_name()
        assembler = AssemblyWrapperNames.get_assembly_tool(assembler_name)

        # use the assembler to assemble the reads.
        assembly_file_prefix = str(self.pipeline_name) + "_" + "assembly"
        assembly_path = self.directory / assembly_file_prefix
        full_assembly_path = assembler.assemble(
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
        orffinder_name = self.config_object.get_orffinder_name()
        orffinder = OrfFinderWrapperNames.get_orffinding_tool(orffinder_name)
        orf_file_path = orffinder.find_orfs(fasta_path=self.input, outpath=orf_path)

        # get length of each contig/genome
        accession2length = {}
        for contig_name, sequence in FastaUtils.get_proteins(self.input):
            accession2length[contig_name] = len(sequence)

        # turn ORFs into proteins.
        logging.info(f"saving ORF-converted proteins to: {orf_file_prefix_proteins}")
        for orf_name, orf_sequence in FastaUtils.get_proteins(
            orf_file_path, withfullname=True
        ):
            accession_id, start_pos, score = orffinder.get_info_from_name(orf_name)
            stop_pos = start_pos + len(orf_sequence)

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
            model_path = self.config_object.get_model_path(model)
            model_predictor_name = self.config_object.get_predictor_model_name(model)
            model_object = ModelNames.get_model(model_predictor_name)
            model_object = model_object.load(model_path)

            # get the model classes.
            model_classes_csv = self.config_object.get_model_classes(model)
            with open(model_classes_csv, "r") as f:
                index2class = list(csv.DictReader(f))[-1]  # get latest.
                del index2class["datetime"]
                index2class = {int(k): v for k, v in index2class.items()}

            # predict features from proteins.
            x_test = np.vstack(self.dataframe["features"].to_numpy())
            self.dataframe[model] = model_object.predict(x_test)

            # replace values with class names.
            self.dataframe[model].replace(index2class, inplace=True)

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
