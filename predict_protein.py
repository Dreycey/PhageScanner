import logging
import os
from pathlib import Path
from typing import Dict, Union
import gc
import sys

import numpy as np
import pandas as pd
import swifter

from PhageScanner.main import utils
from PhageScanner.main.exceptions import IncorrectValueError
from PhageScanner.main.feature_extractors import ProteinFeatureExtraction, DPCExtractor, ProteinFeatureAggregator
from PhageScanner.main.models import ModelNames
from PhageScanner.main.pipelines.pipeline_interface import Pipeline


protein = sys.argv[1]
path2savemodel = Path(sys.argv[2])

#extract features
#protein = ProteinFeatureExtraction.clean_protein(protein)
dpc_extractor = DPCExtractor({"gap_size" : 0})
protein_feature_aggregator = ProteinFeatureAggregator([dpc_extractor])
feature_vector = protein_feature_aggregator.extract_features(protein)
feature_vector = np.vstack([feature_vector])

with open("feature_vector", "w") as f:
    print(feature_vector)
    f.write(f"{feature_vector}")

# call model
print(feature_vector.shape)
model_object = ModelNames.get_model("FFNN")
new_model = model_object.load(path2savemodel)
predictions, _  = new_model.predict(feature_vector)

print(predictions)



