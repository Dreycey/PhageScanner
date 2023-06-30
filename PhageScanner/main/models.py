""" This module contains the models.

Description:
    This module contains the models that are trained
    during the training pipeline. The configuration
    yaml file contains the name of the corresponding model.
    This class also contains methods for evaluating the
    models performance.
"""
import logging
import os
import random
import shutil
import tempfile
import time
from abc import ABC, abstractclassmethod, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List

import joblib
import numpy as np
from keras.layers import (
    LSTM,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling1D,
)
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l1

# scikit-learn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# in-house libraries
from PhageScanner.main.blast_wrapper import BLASTWrapper
from PhageScanner.main.exceptions import (
    IncorrectValueError,
    IncorrectYamlError,
    MissingFileError,
)
from PhageScanner.main.utils import FastaUtils



class ModelNames(Enum):
    """Names of onboarded models.

    Description:
        This enum contains the names of different models
        that are used for training, testing and application.
    """

    svm = "SVM"
    ffnn = "FFNN"
    multinaivebayes = "MULTINAIVEBAYES"
    gradboost = "GRADBOOST"
    randomforest = "RANDOMFOREST"
    blast = "BLAST"
    logreg = "LOGREG"
    cnn = "CNN"
    rnn = "RNN"

    @classmethod
    def get_model(cls, name):
        """Return the the corresponding cluster adapter (Factory-like pattern)"""
        name2adapter = {
            cls.svm.value: SVCMultiClassModel(),
            cls.ffnn.value: FFNNMultiClassModel(),
            cls.multinaivebayes.value: MultiNaiveBayesClassModel(),
            cls.gradboost.value: GradientBoostingClassModel(),
            cls.randomforest.value: RandomForestClassModel(),
            cls.blast.value: BlastClassifier(),
            cls.logreg.value: LogRegClassModel(),
            cls.cnn.value: CNNMultiClassifier(),
            cls.rnn.value: RNNMultiClassifier(),
        }
        adapter = name2adapter.get(name)

        if adapter is None:
            tools_available = ",".join(name2adapter.keys())
            exception_string = (
                "The Clustering tool requested in the Yaml File is not available. "
            )
            exception_string += f"The requested tool in the Yaml is: {name}. "
            exception_string += f"The options available are: {tools_available}"
            raise IncorrectYamlError(exception_string)
        return adapter


class Model(ABC):
    """This class defines the template for a model.

    Description:
        This class defines the basic template for any
        model used within the pipeline.
    """

    @abstractmethod
    def train(self, x, y):
        """Train a binary classification model."""
        pass

    @abstractmethod
    def predict(self, x, y):
        """Use the model to predict a binary class."""
        pass

    @abstractmethod
    def save(self, file_path: Path):
        """Save a model to disk."""
        pass

    @classmethod
    @abstractclassmethod
    def load(cls, file_path: Path):
        """Load a model from disk"""
        pass

    def vectorize(self, index_array):
        """Turn a 1D array of indexes into a 2D array.

        Description:
            This function takes a 1D array of indexes
            and converts it into a 2D vector of the
            1 hot encoded indexes.

        Example:
            [2,1,1,0] -> [[0,0,1], [0,1,0], [0,1,0], [1,0,0]]
        """
        vec = np.zeros(len(set(index_array)), len(index_array))
        for row, col in enumerate(index_array):
            vec[row][col] = 1
        return vec

    def test(self, test_x, test_y):
        """Test the model on known data."""
        # get predictions.
        start_time = time.time()
        predictions = self.predict(test_x)
        execution_time = time.time() - start_time

        # create output dictionary
        result_dictionary = {
            "accuracy": np.round(accuracy_score(predictions, test_y), 3),
            "confusion_matrix": confusion_matrix(predictions, test_y),
            "f1score": np.round(f1_score(predictions, test_y, average=None), 3),
            "precision": np.round(
                precision_score(predictions, test_y, average=None), 3
            ),
            "recall": np.round(recall_score(predictions, test_y, average=None), 3),
            "execution_time_seconds": round(execution_time, 6),
        }

        return result_dictionary


class ScikitModel(Model):
    """This class defines the the base class for Scikit models.

    Description:
        This class defines the basic template for all
        models using the Scikit-learn API.
    """

    file_extension = ".joblib"

    def save(self, file_path: Path):
        """Save a model to disk for scikit learn."""
        joblib.dump(self.model, str(file_path) + self.file_extension)

    @classmethod
    def load(cls, file_path: Path):
        """Load a model from disk for scikitlearn."""
        model_obj = cls()
        model_obj.model = joblib.load(str(file_path) + cls.file_extension)
        return model_obj

    def train(self, train_x, train_y):
        """Train the model."""
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        """Predict a classes for an array of input proteins."""
        predictions = self.model.predict(test_x)
        return predictions


class KerasModel(Model):
    """This class defines the the base class for Keras models.

    Description:
        This class defines the basic template for all
        models using the Keras API.
    """

    file_extension = ".h5"

    def save(self, file_path: Path):
        """Save a model to disk for scikit learn."""
        self.model.save(str(file_path))

    @classmethod
    def load(cls, file_path: Path):
        """Load a model from disk for scikitlearn."""
        model_obj = cls()
        model_obj.model = load_model(str(file_path))
        return model_obj

    def predict(self, test_x):
        """Predict a classes for an array of input proteins."""
        prediction_probabilities = self.model.predict(test_x)
        predictions = np.argmax(prediction_probabilities, axis=-1)
        return predictions


class SVCMultiClassModel(ScikitModel):
    """Class for SVM/SVC model."""

    def __init__(self):
        """Instantiate a new SVCMultiClassModel."""
        self.model = make_pipeline(StandardScaler(), SVC(random_state=0, tol=1e-5))


class MultiNaiveBayesClassModel(ScikitModel):
    """Class for multinomial naive bayes model."""

    def __init__(self):
        """Instantiate a new MultiNaiveBayesClassModel."""
        self.model = MultinomialNB(force_alpha=True)


class LogRegClassModel(ScikitModel):
    """Class for logistic regression one-verse-all model."""

    def __init__(self):
        """Instantiate a new LogRegClassModel."""
        self.model = LogisticRegression(random_state=0, multi_class="ovr")


class GradientBoostingClassModel(ScikitModel):
    """Class for gradient boosting model."""

    def __init__(self):
        """Instantiate a new GradientBoostingClassModel."""
        self.model = GradientBoostingClassifier(
            n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0
        )


class RandomForestClassModel(ScikitModel):
    """Class for random forest model."""

    def __init__(self):
        """Instantiate a new RandomForestClassModel."""
        self.model = RandomForestClassifier(max_depth=10, random_state=0)


class FFNNMultiClassModel(KerasModel):
    """Class defining the PhANNs FFNN."""

    def __init__(self):
        """Instantiate the FFNN model."""
        self.model = None

    def build_model(self, feature_vector_length, number_of_classes):
        """Build the model from the feature vector length."""
        logging.info(
            f"Building FFNN model. Feature vector length: {feature_vector_length}"
        )
        logging.info(f"Building FFNN model. Number of classes: {number_of_classes}")

        # initialize the constructor
        model = Sequential()
        model.add(
            Dense(
                100,
                activation="relu",
                input_shape=(feature_vector_length,),
                kernel_initializer="random_uniform",
            )
        )

        # hidden layers
        model.add(Dropout(0.2))
        model.add(Dense(200, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(200, activation="relu"))
        model.add(Dropout(0.2))

        # Add an output layer
        model.add(Dense(number_of_classes, activation="softmax"))

        # compile the model
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
        model.compile(
            loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

        return model

    def train(self, train_x, train_y):
        """Train an FFNN on multiclass data"""
        if self.model is None:
            self.model = self.build_model(
                feature_vector_length=len(train_x[0]),
                number_of_classes=max(train_y) + 1,
            )

        self.model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=1)


class RNNMultiClassifier(KerasModel):
    """RNN MultiClassifier built"""

    def __init__(self):
        """Initialize the RNN MultiClassifier."""
        self.model = None

    def build_model(self, row_length, column_length, number_of_classes):
        """Build the RNN Model.

        Description:
            Sets the layers and parameters for the RNN. The last functionality
            of this method is comiling the model.
        """
        # Create a sequential model
        model = Sequential()

        # Add an LSTM layer
        model.add(
            LSTM(100, input_shape=(row_length, column_length), return_sequences=False)
        )

        # add FF layers
        model.add(Dense(1000, activation="relu"))
        model.add(Dense(100, activation="relu"))

        # last, output, layer
        model.add(Dense(number_of_classes, activation="softmax"))

        # Compile the model
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        return model

    def train(self, train_x, train_y):
        """Train an RNN on multiclass data"""
        if self.model is None:
            self.model = self.build_model(
                row_length=len(train_x[0]),
                column_length=len(train_x[0][0]),
                number_of_classes=max(train_y) + 1,
            )

        self.model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=1)


class CNNMultiClassifier(KerasModel):
    """CNN MultiClassifier built using the outline of DeepPVP."""

    def __init__(self):
        """Construct the CNN."""
        self.model = None

    def build_model(self, row_length, column_length, number_of_classes):
        """Build the CNN.

        Description:
            Sets the layers and parameters for the CNN. The last functionality
            of this method is comiling the model.
        """
        # Define the model
        model = Sequential()
        # Convolutional layer
        model.add(
            Conv1D(
                filters=32,
                kernel_size=3,
                activation="relu",
                input_shape=(row_length, column_length),
            )
        )
        # Max pooling layer
        model.add(MaxPooling1D(pool_size=4))
        # Batch normalization layer
        model.add(BatchNormalization())
        # Flatten the output from the previous layer
        model.add(Flatten())
        # Fully connected layer
        model.add(Dense(64, activation="relu", kernel_regularizer=l1(0.01)))
        # Compile the model
        # if number_of_classes > 2:
        model.add(Dense(number_of_classes, activation="softmax"))
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        return model

    def train(self, train_x, train_y):
        """Train an CNN on multiclass data"""
        if self.model is None:
            self.model = self.build_model(
                row_length=len(train_x[0]),
                column_length=len(train_x[0][0]),
                number_of_classes=max(train_y) + 1,
            )

        self.model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=1)


class BlastClassifier(BLASTWrapper, Model):
    """Creates a classifier around BLAST."""

    def __init__(self, database_path=None):
        """Construct the BLAST classifier."""
        self.makedbcmd = "makeblastdb"
        self.querycmd = "blastp"
        self.dbpath = database_path
        self.temp_directory = None

    def __del__(self):
        """Delete the temporary directory, if it exists."""
        if self.temp_directory is not None and os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)

    def save(self, file_path: Path):
        """Save the blast database in a particular location.

        Description:
            For blast, this means *copying* the files to a new location.
            So:
                1. Make the new directory.
                2. Move files.
                3. Update DB path
        """
        # 1. Make the new directory.
        os.makedirs(file_path, exist_ok=True)

        # 2. Move files.
        for filename in os.listdir(self.dbpath.parent):
            # Construct full file paths
            source = os.path.join(self.dbpath.parent, filename)
            target = os.path.join(file_path, filename)

            # Move file to target directory
            shutil.copy(source, target)

        # 3. Update DB path
        self.dbpath = (file_path,)

    @classmethod
    def load(cls, file_path: Path):
        """Create a new blast classifier class.

        Description:
            The only thing this really does is tell
            blast where the database is located.
        """
        model_obj = cls(database_path=file_path / "BLAST_DB")
        return model_obj

    def train(self, train_x: List[str], train_y: List[int]):
        """Create a blast database using model template.

        Description:
            This method creates a blast database using the same
            input that other models/classifiers recieve. This
            allows for treating blast identical to other classifiers.

        Notes:
            1.  This method creates a blast database using a temporary
                directory that will contain the blast files.
        """
        # create a local temp fasta file for creating the database.
        temp_fasta_path = Path(tempfile.NamedTemporaryFile(delete=True).name)
        self._save_array_to_fasta(
            array=train_x, output_file=temp_fasta_path, classes=train_y
        )

        # ensure proteins are saved correctly.
        seq_count_in_temp_fasta = FastaUtils.count_entries_in_fasta(temp_fasta_path)
        seq_count_expected = len(train_x)
        if seq_count_in_temp_fasta != seq_count_expected:
            err_message = "(Blast Model) Fasta file does not contain "
            err_message += f"all proteins sequences: {temp_fasta_path}"
            err_message += f"Should have {seq_count_expected} sequences "
            err_message += f"but only has {seq_count_in_temp_fasta}"
            raise MissingFileError(err_message)

        # create the blast database.
        self.temp_directory = tempfile.mkdtemp()
        logging.debug(f"Creating blast database at `{self.temp_directory}`..")
        self.dbpath = Path(self.temp_directory) / "BLAST_DB"
        self.create_database(fasta_file=temp_fasta_path, db_name=self.dbpath)
        logging.debug(f"(Finished) Creating blast database at `{self.temp_directory}`")

    def predict(self, test_x: List[str]):
        """Predict the classes of proteins provided as an array.

        Description:
            This function uses Blast as a classifier. The intended
            purpose of blast is not for classifying proteins like this,
            but it is the best way to compare to ML methods. Likewise,
            it does work for this purpose.

        Note:
            1.  The proteins must not be turned into vectorized features!
                The only way to use the module is to use the feature extractor
                that does - well nothing. This allows for BLAST to use the
                proteins in the input to create a local temp fasta file.
            2.  The `self.dbpath` must be defined by this point. This can
                happen by loading or `training` first. An error will be
                raised if not.
        """
        # create a local temp fasta file for testing.
        temp_fasta_path = Path(tempfile.NamedTemporaryFile(delete=True).name)
        self._save_array_to_fasta(array=test_x, output_file=temp_fasta_path)

        # ensure proteins are saved correctly.
        seq_count_in_temp_fasta = FastaUtils.count_entries_in_fasta(temp_fasta_path)
        seq_count_expected = len(test_x)
        if seq_count_in_temp_fasta != seq_count_expected:
            err_message = "(Blast Model) Fasta file does not contain "
            err_message += f"all proteins sequences: {temp_fasta_path}. "
            err_message += f"Should have {seq_count_expected} sequences "
            err_message += f"but only has {seq_count_in_temp_fasta}"
            raise MissingFileError(err_message)

        # use the local fasta file to create the blast database.
        logging.debug("Getting predictions using BLAST..")
        prediction = self._get_classifications(temp_fasta_path)
        logging.debug("(Finished) Getting predictions using BLAST!")

        # raise an error if the length(predictions) != len(input proteins)
        if len(prediction) != len(test_x):
            error_message = "The length of the predicted classes is "
            error_message += "not equal to the number of input proteins! "
            error_message += (
                f"predicted classes: {len(prediction)}, proteins: {len(test_x)}"
            )
            raise IncorrectValueError(error_message)

        return prediction

    def _save_array_to_fasta(
        self, array: List[str], output_file: Path, classes: List[int] = None
    ):
        """Save the array to a local output file.

        Description:
            This is needed because the blast command line tool
            uses local fasta files files for queries and for
            building the blast database.
        """
        logging.debug(f"Creating fasta from array here: `{output_file}`..")
        with open(output_file, "a") as output_fasta:
            # get the name for the protein.
            # NOTE: If building DB, classes should be an array of the
            #       corresponding class index. (i.e. List[int])
            for index, protein in enumerate(array):
                if classes is not None:
                    output_name = classes[index]
                else:
                    output_name = index
                # get protein from the 1D vector format.
                protein = protein[0]
                # add protein and name to fasta.
                output_fasta.write(f">{output_name}\n{protein}\n")

        # ensure proteins are saved correctly.
        seq_count_in_temp_fasta = FastaUtils.count_entries_in_fasta(output_file)
        seq_count_expected = len(array)
        if seq_count_in_temp_fasta != seq_count_expected:
            err_message = "(Blast Model) Fasta file does not contain "
            err_message += f"all proteins sequences: {output_file}"
            err_message += f"Should have {seq_count_expected} sequences "
            err_message += f"but only has {seq_count_in_temp_fasta}"
            raise MissingFileError(err_message)

        logging.debug(f"(Finished) Creating fasta from array here: `{output_file}`")

    def _get_classifications(self, fasta_file: Path, threads: int = 1):
        """Return the classifications.

        Description:
            Returns the classifications of the protein
            given an array of proteins.

        Note:
            1.  The `self.dbpath` must be defined by this point. This can
                happen by loading or `training` first. An error will be
                raised if not.
        """
        # name of temporary file for output.
        blast_temp_file = Path(tempfile.NamedTemporaryFile(delete=False).name)

        # query blast db.
        self.query(fasta_file=fasta_file, outputfile=blast_temp_file, threads=threads)

        # ensure that blast produced an output file.
        if not os.path.exists(blast_temp_file):
            error_message = "The output blast file does not exist "
            error_message += f"Here is the location (temp): `{blast_temp_file}`"
            raise MissingFileError(error_message)
        elif os.path.getsize(blast_temp_file) == 0:
            error_message = "The output blast file is empty but exists. "
            error_message += f"Here is the location (temp): `{blast_temp_file}`"
            raise MissingFileError(error_message)

        # open the output file and find classes.
        output_classes = self._parse_blast_results(blast_temp_file)
        classes_guessed = list(set(output_classes.values()))

        # for each accession in the fasta, find the class/randomly choose
        prediction_array = []
        for accession, _ in FastaUtils.get_proteins(fasta_file):
            if accession in output_classes:
                prediction_array.append(output_classes[accession])
            else:
                logging.warning("Could not BLAST protein! Guessing the class randomly.")
                prediction_array.append(random.choice(classes_guessed))

        return np.array(prediction_array)

    def _parse_blast_results(self, results: Path):
        """Parse the Blast output results.

        Description:
            Parses the blast results, returning a dictionary
            mapping each accession ID to the blast classification.
        """
        output_classes = {}
        current_accesion = None
        top_score, current_class = 0, None
        with open(results, "r") as temp_file:
            for line in temp_file.readlines():
                # parse the new line in the output file.
                accession, class_name, score = line.strip("\n").split("\t")
                score, class_name = int(score), int(class_name)

                # find top class for each protein.
                if current_accesion == accession:
                    if top_score < score:
                        top_score = score
                        current_class = class_name
                else:
                    if current_accesion and (current_class is not None):
                        output_classes[current_accesion] = current_class
                    # update values for the new protein.
                    current_accesion = accession
                    top_score = score
                    current_class = class_name
            # last value
            output_classes[current_accesion] = current_class

        return output_classes
