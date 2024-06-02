"""The training pipeline trains models for prediction protein classes."""

import logging
import os
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from PhageScanner.main import utils
from PhageScanner.main.exceptions import IncorrectValueError
from PhageScanner.main.feature_extractors import ProteinFeatureExtraction
from PhageScanner.main.models import ModelNames
from PhageScanner.main.pipelines.pipeline_interface import Pipeline


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
        """Balance classes  within each partition.

        Description:
            Here we upsample the smaller classes with replacement. The
            justification is that there may be large negative classes,
            for example non-pvp, that contain more unique proteins. For the
            models to learn these, we don't want to downsample. Likewise, if
            there are really small classes, then the models would lose context
            on other potential proteins within the other classes.
        """
        new_balanced_partitions_df = []
        for partition in self.dataframe["partition"].unique():
            partition_df = self.dataframe[self.dataframe["partition"] == partition]

            # get smallest class size.
            class_sizes = partition_df["class"].value_counts()
            max_class_size = max(class_sizes)

            # create a balanced partition.
            balanced_partition_df_list = []
            for class_index in partition_df["class"].unique():
                # get proteins corresponding to the  class index.
                class_df: pd.DataFrame = partition_df[
                    partition_df["class"] == class_index
                ]

                # add to balanced_partition_df_list
                balanced_partition_df_list.append(class_df)

                # get differrence from the max class size, then upsample w/ replacement
                size_difference = max_class_size - len(class_df)
                if size_difference > 0:
                    balanced_class_df = class_df.sample(
                        n=size_difference, random_state=1, replace=True
                    )
                    balanced_partition_df_list.append(balanced_class_df)

            # combine balanced classes for this partition.
            balanced_partition_df = pd.concat(
                balanced_partition_df_list, ignore_index=True
            )

            # ensure the min class size is now the same as max.
            assert max_class_size == min(balanced_partition_df["class"].value_counts())

            # add balanced partition to new_balanced_partitions_df.
            new_balanced_partitions_df.append(balanced_partition_df)

        # combine all balanced partitions to a single dataframe.
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
        utils.CSVUtils.appendcsv(
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
        utils.CSVUtils.appendcsv(
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

        # Step 2: clean proteins
        logging.info("Step 2 - Cleaning proteins...")
        self.dataframe["protein"] = self.dataframe["protein"].apply(
            ProteinFeatureExtraction.clean_protein
        )
        logging.info("Step 2 (Finished) - Cleaning proteins...")

        # Step 3: Balance the classes with different partitions.
        logging.info("Step 3 - Balancing classes in each partition...")
        self.balance_partitions()
        logging.info("Step 3 (Finished) - Balancing classes in each partition...")

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
