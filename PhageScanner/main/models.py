""" This module contains the models.

Description:
    This module contains the models that are trained
    during the training pipeline. The configuration
    yaml file contains the name of the corresponding model.
    This class also contains methods for evaluating the
    models performance.
"""
import logging
import time
from abc import ABC, abstractclassmethod, abstractmethod
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
from keras.layers import Dense, Dropout
# FFNN
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# scikit-learn models
# SVM model
from sklearn.svm import SVC

# in-house libraries
from PhageScanner.main.exceptions import IncorrectYamlError


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

    @classmethod
    def get_model(cls, name):
        """Return the the corresponding cluster adapter (Factory-like pattern)"""
        name2adapter = {
            cls.svm.value: SVCMultiClassModel(),
            cls.ffnn.value: FFNNMultiClassModel(),
            cls.multinaivebayes.value: MultiNaiveBayesClassModel(),
            cls.gradboost.value: GradientBoostingClassModel(),
            cls.randomforest.value: RandomForestClassModel(),
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
            "f1score": np.round(f1_score(predictions, test_y, average="weighted"), 3),
            "precision": np.round(
                precision_score(predictions, test_y, average="weighted"), 3
            ),
            "recall": np.round(
                recall_score(predictions, test_y, average="weighted"), 3
            ),
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
