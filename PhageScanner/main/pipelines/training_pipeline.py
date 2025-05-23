"""The training pipeline trains models for prediction protein classes."""

import logging
import os
from pathlib import Path
from typing import Dict, Union, Iterator, Optional, Tuple, Any
import gc
from enum import Enum

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

from PhageScanner.main import utils
from PhageScanner.main.exceptions import IncorrectValueError
from PhageScanner.main.feature_extractors import ProteinFeatureExtraction, extract_feature_vector
from PhageScanner.main.models import ModelNames
from PhageScanner.main.pipelines.pipeline_interface import Pipeline


class DataframeColumns(str, Enum):
    CLASS_LABEL = "class"
    PARTITION_LABEL = "partition"
    PROTEIN_SEQ = "protein"


class TrainingPipeline(Pipeline):
    """A pipeline for training and testing ML models.

    Description:
        This class controls the training and testing
        of machine learning models specified in the
        input config.
    """

    def __init__(self, config: Path, db_directory: Path, directory: Path):
        """Initialize the training pipeline."""
        logging.info("Running TrainingPipeline | Creating pipeline...")
        self.config_object = utils.TrainingConfig(config)
        self.directory = directory
        self.number_of_classes = 0  # updated during pipeline run.
        self.db_directory = db_directory

        # create directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
            
        # Create class name to index mapping.
        model_classes = self.config_object.get_classes()
        self.number_of_classes = len(model_classes)
        
        # Create class name to index mapping.
        self.class2index = dict()
        self.class2filepath = dict()
        for index, class_name in enumerate(model_classes):
            class_name = class_name.get("name")
            class_path = self.config_object.get_csv_path_from_name(
                class_name, self.db_directory
            )
            self.class2index[class_name] = index
            self.class2filepath[class_name] = class_path
            
        # Get model information
        models = self.config_object.get_models()
        self.modelname2information = dict()
        for model_name in models:
            self.modelname2information[model_name] = {
                'path2savemodel' : self.directory / f"{model_name}",
                'model_features' : [i for i in self.config_object.get_model_features(model_name)],
                'segement_size' : self.config_object.sequential(model_name),
                'model_predictor_name' : self.config_object.get_predictor_model_name(model_name)       
            }

    @staticmethod
    def create_combined_df(class2filepath, class2index):
        """Create a Pandas dataframe from the combined partitions for different classes CSVs."""
        dataframes = []

        for class_name, class_path in class2filepath.items():
            logging.debug(f"Combining class {class_name} into the dataframe.")
            df_tmp = pd.read_csv(class_path)
            df_tmp[DataframeColumns.CLASS_LABEL] = class2index[class_name]
            dataframes.append(df_tmp)

        return pd.concat(dataframes, ignore_index=True)

    @staticmethod
    def hybrid_sample_partitions(df: pd.DataFrame, percentile: float = 50, interpolation: str = 'linear') -> pd.DataFrame:
        """
        Balance classes within each partition by upsampling smaller classes and 
        downsampling larger classes based on an interpolated percentile.
        """
        balanced_partitions = []

        for _, partition_df in df.groupby(DataframeColumns.PARTITION_LABEL):
            class_sizes = partition_df[DataframeColumns.CLASS_LABEL].value_counts()
            percentile_class_size = int(np.percentile(class_sizes, percentile, interpolation=interpolation))

            balanced_classes = []
            for _, class_df in partition_df.groupby(DataframeColumns.CLASS_LABEL):
                if class_df.count() > percentile_class_size:
                    sampled_df = class_df.sample(n=percentile_class_size, random_state=1)
                else:
                    sampled_df = class_df.sample(n=percentile_class_size, replace=True, random_state=1)
                balanced_classes.append(sampled_df)

            balanced_partitions.append(pd.concat(balanced_classes, ignore_index=True))

        return pd.concat(balanced_partitions, ignore_index=True)
        
    def train_and_test_models(self, df, modelname2information):
        """ Perform model training and testing. """
        for model_name, model_information in modelname2information.items():
            kfold_iteration = 0
            path2savemodel, model_features, segement_size, model_predictor_name = (model_information['path2savemodel'],
                                                                                   model_information['model_features'],
                                                                                   model_information['segement_size'],
                                                                                   model_information['model_predictor_name'])
            for x_train, y_train, x_test, y_test in TrainingPipeline.get_kfold_training(df, model_features, segement_size):
                logging.info(f"Training and testing model: {model_name} (Iteration {kfold_iteration})...")

                model_object = ModelNames.get_model(model_predictor_name)
                model_object.train(x_train, y_train, x_test, y_test)
                model_object.save(path2savemodel)
                model_results = model_object.test(x_test, y_test)

                TrainingPipeline.save_model_results(
                    directory_to_save_results=self.directory,
                    model_name=model_name,
                    model_features=model_features,
                    pipeline_start_time=self.pipeline_start_time,
                    iteration=kfold_iteration,
                    model_results=model_results,
                )

                # increment the kfold iteration
                kfold_iteration += 1
                # free up memory
                del x_train, y_train, x_test, y_test
                gc.collect()
                
                logging.info(f"Finished Training and testing Model: {model_name} (Iteration {kfold_iteration})...")

    @staticmethod
    def get_kfold_training(df, model_features, segment_size):
        """Get X_train, Y_train, X_test, Y_test.

        Description:
            Obtain training data split into sets that allow
            for training and testing to be performed based
            on how the data is partitioned.
        """
        unique_partitions = df["partition"].unique()
        for test_partition in unique_partitions:
            train_df = df[df["partition"] != test_partition]
            test_df = df[df["partition"] == test_partition]

            x_train = extract_feature_vector(train_df[DataframeColumns.PROTEIN_SEQ].to_numpy(),
                                             model_features,
                                             segment_size)
            x_test = extract_feature_vector(test_df[DataframeColumns.PROTEIN_SEQ].to_numpy(),
                                            model_features, 
                                            segment_size)

            y_train = train_df[DataframeColumns.CLASS_LABEL].to_numpy()
            y_test = test_df[DataframeColumns.CLASS_LABEL].to_numpy()

            yield x_train, y_train, x_test, y_test

    @staticmethod
    def save_model_results(
        directory_to_save_results: Path,
        model_name: str,
        model_features: Iterator[Tuple[str, Optional[Any]]],
        pipeline_start_time,
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
            directory_to_save_results / f"{model_name}_confusion_matrix.csv"
        )
        np.savetxt(
            confusion_matrix_output_path,
            model_results["confusion_matrix"],
            delimiter=",",
            fmt="%.3f",
        )
        del model_results["confusion_matrix"]

        # Save the f1score, accuracy, and precision
        output_path = directory_to_save_results / "model_results.csv"
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
        for feature, params in model_features:
            if params:
                param_strs = [f"{k}={v}" for k, v in params.items()]
                params_text = "; ".join(param_strs)
                feature_string = f"{feature} ({params_text})"
            else:
                feature_string = feature
            features.append(feature_string)

        # add columns not originally present in the dataframe
        model_results.update({
            "datetime": pipeline_start_time,
            "model": model_name,
            "kfold_iteration": iteration,
            "features": features
        })

        # convert columns with many entries to strings
        for key in ["f1score", "precision", "recall", "features"]:
            model_results[key] = "\t".join(map(str, model_results[key]))

        # save results to disk.
        utils.CSVUtils.appendcsv(
            data_dict=[model_results],  # input must be an array.
            fieldnames=csv_file_headers,
            file_path=output_path,
        )

    def run(self):
        """Run the training pipeline."""
        # Step 1: Combine partitions from different classes.
        logging.info("Step 1 - Combine partitions...")
        combined_dataframe = TrainingPipeline.create_combined_df(self.class2filepath, self.class2index)
        logging.info("Step 1 (Finished) - Combine partitions...")

        # Step 2: Clean proteins
        logging.info("Step 2 - Cleaning proteins...")
        combined_dataframe["protein"] = combined_dataframe["protein"].apply(
            ProteinFeatureExtraction.clean_protein
        )
        logging.info("Step 2 (Finished) - Cleaning proteins...")

        # Step 3: Balance the classes with different partitions.
        logging.info("Step 3 - Balancing classes in each partition...")
        combined_dataframe = TrainingPipeline.hybrid_sample_partitions(combined_dataframe, percentile=70)
        logging.info("Step 3 (Finished) - Balancing classes in each partition...")

        # Step 4: Extract features and train.
        logging.info("Step 4 - Training and Testing Models")
        self.train_and_test_models(combined_dataframe, self.modelname2information)
        logging.info("Step 4 (Finished) - Training and Testing Models")