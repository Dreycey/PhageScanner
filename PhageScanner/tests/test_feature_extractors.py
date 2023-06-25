"""Module for testing feature extraction methods.

Description:
    This module tests each feature extraction method
    to ensure that the feature extraction is working
    as expected.
"""
import unittest

import numpy as np

from PhageScanner.main.exceptions import IncorrectValueError, ProteinError
from PhageScanner.main.feature_extractors import (AACExtractor, ATCExtractor,
                                                  DPCExtractor,
                                                  ProteinFeatureAggregator)


class TestATCExtractor(unittest.TestCase):
    """Test a feature extraction method for ATC."""

    def setUp(self):
        """Set up a test instance of ATCExtractor."""
        self.extractor = ATCExtractor()
        self.extractor.amino_acid_atom_counts = {
            "M": {"C": 5, "H": 9, "N": 1, "O": 1, "S": 1},  # Methionine
            "A": {"C": 3, "H": 5, "N": 1, "O": 1, "S": 0},  # Alanine
        }

    def test_init(self):
        """Test the __init__ method."""
        expected_atom2index = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4}
        self.assertEqual(self.extractor.atom2index, expected_atom2index)

    def test_extract_features(self):
        """Test the extract_features method."""
        protein = "MAM"
        result = self.extractor.extract_features(protein)
        total_atoms = 2 * (5 + 9 + 1 + 1 + 1) + (3 + 5 + 1 + 1)
        expected_result = np.array(
            [
                13 / total_atoms,
                23 / total_atoms,
                3 / total_atoms,
                3 / total_atoms,
                2 / total_atoms,
            ]
        )
        np.testing.assert_almost_equal(result, expected_result, decimal=7)


class TestAACExtractor(unittest.TestCase):
    """Tests the AAC extraction method."""

    def setUp(self):
        """Set up a test instance of AACExtractor."""
        self.extractor = AACExtractor()
        self.extractor.canonical_amino_acids = self.extractor.canonical_amino_acids

    def test_init(self):
        """Test the __init__ method."""
        expected_aminoacid2index = {
            aa: index for index, aa in enumerate(self.extractor.canonical_amino_acids)
        }
        self.assertEqual(self.extractor.aminoacid2index, expected_aminoacid2index)

    def test_extract_features(self):
        """Test the extract_features method."""
        protein = "ACDEFGHIKLMNPQRSTVWY"
        result = self.extractor.extract_features(protein)
        expected_result = np.ones(len(self.extractor.canonical_amino_acids)) / len(
            protein
        )
        np.testing.assert_almost_equal(result, expected_result, decimal=7)


class TestDPCExtractor(unittest.TestCase):
    """Class for testing the DPC extract method."""

    def setUp(self):
        """Set up a test instance of DPCExtractor."""
        self.extractor = DPCExtractor(parameters={"gap_size": 0})
        self.extractor.canonical_amino_acids = self.extractor.canonical_amino_acids

    def test_init(self):
        """Test the __init__ method."""
        expected_aa2index = {}
        index = 0
        for aa1 in self.extractor.canonical_amino_acids:
            for aa2 in self.extractor.canonical_amino_acids:
                expected_aa2index[aa1 + aa2] = index
                index += 1
        self.assertEqual(self.extractor.aa2index, expected_aa2index)

    def test_extract_features(self):
        """Test the extract_features method."""
        protein = "ACDEFGHIKLMNPQRSTVWY"
        result = self.extractor.extract_features(protein)
        expected_result = np.zeros(len(self.extractor.canonical_amino_acids) ** 2)
        for i in range(len(protein) - 1):
            dipeptide = protein[i] + protein[i + 1]
            index = self.extractor.aa2index[dipeptide]
            expected_result[index] += 1 / (len(protein) - 1)
        np.testing.assert_almost_equal(result, expected_result, decimal=7)

    def test_incorrect_value_error(self):
        """Test that an error is raised when parameters is None."""
        with self.assertRaises(IncorrectValueError):
            DPCExtractor(parameters=None)

    def test_protein_error(self):
        """Test that an error is raised when the protein is too short."""
        extractor = DPCExtractor(parameters={"gap_size": 5})
        extractor.canonical_amino_acids = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]
        with self.assertRaises(ProteinError):
            extractor.extract_features("ACDE")


class TestProteinFeatureAggregator(unittest.TestCase):
    """Class for testing the feature aggregator."""

    def setUp(self):
        """Set up a test instance of ProteinFeatureAggregator."""
        self.extractors = [
            AACExtractor(),
            ATCExtractor(),
            DPCExtractor(parameters={"gap_size": 0}),
        ]
        self.aggregator = ProteinFeatureAggregator(self.extractors)

        # test init
        self.assertEqual(self.aggregator.extractors, self.extractors)

    def test_extract_features(self):
        """Test the extract_features method."""
        protein = "ACDEFGHIKLMNPQRSTVWY"
        result = self.aggregator.extract_features(protein)
        expected_result = np.hstack(
            [extractor.extract_features(protein) for extractor in self.extractors]
        )
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_proteins_viewed(self):
        """Test the proteins_viewed counter."""
        protein = "ACDEFGHIKLMNPQRSTVWY"
        proteins_to_view = 10
        self.aggregator.proteins_viewed = 0  # reset the counter
        for _ in range(proteins_to_view):
            self.aggregator.extract_features(protein)
        self.assertEqual(self.aggregator.proteins_viewed, proteins_to_view)
