"""This pipeline interface enforces each inheriting class impliments the run method."""

import logging
import time
from abc import ABC, abstractmethod

from PhageScanner.main.feature_extractors import (
    FeatureExtractorNames,
    ProteinFeatureAggregator,
    SequentialProteinFeatureAggregator,
)


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
        logging.info(f"extracting protein features: '{model_name}'")

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
        self.dataframe["features"] = self.dataframe["protein"].apply(
            aggregator.extract_features
        )

        logging.info(f"done extracting features for model: '{model_name}'")
