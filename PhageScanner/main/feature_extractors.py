"""Module contains classes for feature extraction on proteins.

Description:
    This module contains feature extraction for proteins.

References:
    An excellent review from Chaolu Meng et al. (2020) goes
    over the advances in PVP prediction and highlights
    different feature extraction methods. May of these methods
    are used here.
    DOI: https://doi.org/10.1016/j.bbapap.2020.140406
"""
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from PhageScanner.main.exceptions import (IncorrectValueError,
                                          IncorrectYamlError, ProteinError)
from PhageScanner.third_party.CTD import CalculateCTD
# 3rd party library; great feature extraction methods.
from PhageScanner.third_party.PseudoAAC import _GetPseudoAAC


class FeatureExtractorNames(Enum):
    """Names of feature extraction methods

    Description:
        This enum contains the names of different
        feature extraction methods. It allows for
        passing a string name and returning an
        instantiated class.
    """

    aac = "AAC"
    dpc = "DPC"
    iso = "ISO"
    pseudoaac = "PSEUDOAAC"
    atc = "ATC"
    ctd = "CTD"
    protein_seq = "PROTEINSEQ"

    @classmethod
    def get_extractor(cls, name, parameters: Optional[Dict]):
        """Return the the corresponding feature extractor (Factory pattern)"""
        name2extractor = {
            cls.aac.value: AACExtractor,
            cls.dpc.value: DPCExtractor,
            cls.iso.value: IsoelectricExtractor,
            cls.pseudoaac.value: PseudoAACExtractor,
            cls.atc.value: ATCExtractor,
            cls.ctd.value: CTDExtractor,
            cls.protein_seq.value: ProteinSequenceExtractor,
        }

        # instantiate the class
        extractor_class = name2extractor.get(name, None)

        # log the given features extractor
        logging.debug(f"object {extractor_class} chosen for {name}")

        # raise exception if no match
        if extractor_class is None:
            msg = f"There is no feature: '{name}'! "
            msg += f"use: {list(name2extractor.keys())}"
            raise IncorrectYamlError(msg)

        extractor_obj = extractor_class(parameters)

        return extractor_obj


class ProteinFeatureExtraction(ABC):
    """Abstract base class for all feature extraction methods for proteins."""

    amino_acid_atom_counts = {
        "A": {"C": 3, "H": 5, "N": 1, "O": 1, "S": 0},  # Alanine
        "R": {"C": 6, "H": 12, "N": 4, "O": 1, "S": 0},  # Arginine
        "N": {"C": 4, "H": 6, "N": 2, "O": 2, "S": 0},  # Asparagine
        "D": {"C": 4, "H": 5, "N": 1, "O": 3, "S": 0},  # Aspartic acid
        "C": {"C": 3, "H": 5, "N": 1, "O": 1, "S": 1},  # Cysteine
        "Q": {"C": 5, "H": 8, "N": 2, "O": 2, "S": 0},  # Glutamine
        "E": {"C": 5, "H": 7, "N": 1, "O": 3, "S": 0},  # Glutamic acid
        "G": {"C": 2, "H": 3, "N": 1, "O": 1, "S": 0},  # Glycine
        "H": {"C": 6, "H": 7, "N": 3, "O": 1, "S": 0},  # Histidine
        "I": {"C": 6, "H": 11, "N": 1, "O": 1, "S": 0},  # Isoleucine
        "L": {"C": 6, "H": 11, "N": 1, "O": 1, "S": 0},  # Leucine
        "K": {"C": 6, "H": 12, "N": 2, "O": 1, "S": 0},  # Lysine
        "M": {"C": 5, "H": 9, "N": 1, "O": 1, "S": 1},  # Methionine
        "F": {"C": 9, "H": 9, "N": 1, "O": 1, "S": 0},  # Phenylalanine
        "P": {"C": 5, "H": 7, "N": 1, "O": 1, "S": 0},  # Proline
        "S": {"C": 3, "H": 5, "N": 1, "O": 2, "S": 0},  # Serine
        "T": {"C": 4, "H": 7, "N": 1, "O": 2, "S": 0},  # Threonine
        "W": {"C": 11, "H": 10, "N": 2, "O": 1, "S": 0},  # Tryptophan
        "V": {"C": 5, "H": 9, "N": 1, "O": 1, "S": 0},  # Valine
        "Y": {"C": 9, "H": 9, "N": 1, "O": 2, "S": 0},  # Tyrosine
    }

    canonical_amino_acids = set(sorted(list(amino_acid_atom_counts.keys())))

    @abstractmethod
    def extract_features(self, protein: str):
        """Extract protein features."""
        pass

    @classmethod
    def clean_protein(cls, protein):
        """Convert a protein into conanical form."""
        clean_protein = []
        for aa in protein:
            if aa == "*":  # stop codon
                new_aa = ""
            elif aa not in cls.canonical_amino_acids:
                new_aa = "G"
            else:
                new_aa = aa
            clean_protein.append(new_aa)
        return "".join(clean_protein)


class ProteinSequenceExtractor(ProteinFeatureExtraction):
    """Do nothing - return a protein sequence."""

    def __init__(self, parameters: Optional[Dict] = None):
        """Instantiate AAC extract method."""
        pass

    def extract_features(self, protein: str):
        """Just returns the protein sequence."""
        return protein


class AACExtractor(ProteinFeatureExtraction):
    """Extraction method for Amino Acid Composition (AAC)"""

    def __init__(self, parameters: Optional[Dict] = None):
        """Instantiate AAC extract method."""
        self.aminoacid2index = {
            aa: index for index, aa in enumerate(self.canonical_amino_acids)
        }

    def __eq__(self, other):
        """Set classes equal if same type."""
        if isinstance(other, AACExtractor):
            return True
        return False

    def extract_features(self, protein: str):
        """Obtain amino acid composition.

        Description:
            Counts the number of amino acids in the protein,
            where each amino acid is mapped to a certain index.
            Thereafter this is turned into a frequency.
        """
        # instantiate a new vector
        aac_vec = np.zeros(len(self.canonical_amino_acids))

        # count the number of amino acids in the protein
        for aa in protein:
            index = self.aminoacid2index.get(aa)
            aac_vec[index] += 1

        # obtain the frequency
        for index, aa_count in enumerate(aac_vec):
            aac_vec[index] = aa_count / len(protein)

        return aac_vec


class DPCExtractor(ProteinFeatureExtraction):
    """Extraction method for g-gap Dipeptide composition (DPC)"""

    def __init__(self, parameters: Optional[Dict] = None):
        """Instantiate DPC extract method."""
        self.aa2index = {}

        # raise error if None passed to parameter
        if parameters is None:
            raise IncorrectValueError(
                "DPCExtractor: must have gap_size param! Not None"
            )

        # get specified gap size.
        self.gap_size = parameters.get("gap_size", 0)

        # make map
        index = 0
        for aa1 in self.canonical_amino_acids:
            for aa2 in self.canonical_amino_acids:
                self.aa2index[aa1 + aa2] = index
                index += 1

    def __eq__(self, other):
        """Set classes equal if same type."""
        if isinstance(other, DPCExtractor):
            return True
        return False

    def extract_features(self, protein: str):
        """Extract g-gap Dipeptide composition.

        Description:
            This method counts the frequency of 2 cooccuring
            amino acids within a given protein. When using
            a gap size of 0, it obtains adjacent amino acids.
            If using a gap size above zero, it counts the
            frequency using amino acids gap-size away.
        """
        # ensure protein is long enough.
        if self.gap_size + 1 > len(protein):
            raise ProteinError(
                f"protein {protein} too small for gap size of {self.gap_size}"
            )

        # create vector (len 400)
        dpc_vec = np.zeros(len(self.canonical_amino_acids) ** 2)

        # count cooccuring dipeptides
        aa_p1 = 0
        aa_p2 = aa_p1 + (1 + self.gap_size)
        while aa_p2 < len(protein):
            dipeptide = protein[aa_p1] + protein[aa_p2]
            index = self.aa2index[dipeptide]
            dpc_vec[index] += 1
            aa_p1 += 1
            aa_p2 += 1

        # get frequency
        for index, count in enumerate(dpc_vec):
            dpc_vec[index] = count / (len(protein) - (self.gap_size + 1))

        return dpc_vec


class IsoelectricExtractor(ProteinFeatureExtraction):
    """Extraction method for obtaining an Isoelectric point from a protein"""

    def __init__(self, parameters: Optional[Dict] = None):
        """Instantiate DPC extract method."""
        pass

    def extract_features(self, protein: str):
        """Obtain an isoelectric point using biopython"""
        analysis = ProteinAnalysis(protein)
        return analysis.isoelectric_point()


class ATCExtractor(ProteinFeatureExtraction):
    """Extraction method for obtaining Atomic composition (ATC)"""

    def __init__(self, parameters: Optional[Dict] = None):
        """Initialize the ATC feature extractor."""
        self.atom2index = {}
        for index, atom in enumerate(self.amino_acid_atom_counts["M"].keys()):
            self.atom2index[atom] = index

    def __eq__(self, other):
        """Set classes equal if same type."""
        if isinstance(other, ATCExtractor):
            return True
        return False

    def extract_features(self, protein: str):
        """Obtain the atomic composition for a protein

        Description:
            This metric returns the frequency of the atoms making
            up the amino acids in the given protein sequence.
        """
        atc_vec = np.zeros(len(self.amino_acid_atom_counts["M"]))

        for aa in protein:
            atom_list = self.amino_acid_atom_counts[aa]
            for atom, count in atom_list.items():
                index = self.atom2index[atom]
                atc_vec[index] += count
        return atc_vec / sum(atc_vec)


class CTDExtractor(ProteinFeatureExtraction):
    """Extraction method for obtaining Composition-transition-distribution (CTD)"""

    def __init__(self, parameters: Optional[Dict] = None):
        """Instantiate CTD extract method."""
        pass

    def extract_features(self, protein: str):
        """Obtain the Composition-transition-distribution (CTD) for a protein.

        Description:
            This method returns the Composition-transition-distribution
            for a given protein. Of note, the current implementation here
            uses a 3rd party library for extracting the PseAAC. There are
            several steps to computing these features, so utilizing 3rd
            party libraries is used for all complex feature extraction.

        Note:
            While a 3rd party library is used, this wrapper method is still
            required to ensure the method is consistent with others when used
            later in the ML pipeline.
        """
        return np.array(list(CalculateCTD(protein).values()))


class PseudoAACExtractor(ProteinFeatureExtraction):
    """Extraction method for obtaining Pseudo-Amino Acid Composition (PseAAC)"""

    def __init__(self, parameters: Optional[Dict] = None):
        """Instantiate PseAAC extract method."""
        pass

    def extract_features(self, protein: str):
        """Obtain the Pseudo-Amino Acid Composition (PseAAC) for a protein.

        Description:
            This method calculates the Pseudo-Amino Acid Composition values
            for an input protein. Of note, the current implementation here
            uses a 3rd party library for extracting the PseAAC. There are
            several steps to computing these features, so utilizing 3rd
            party libraries is used for all complex feature extraction.

        Note:
            While a 3rd party library is used, this wrapper method is still
            required to ensure the method is consistent with others when used
            later in the ML pipeline.
        """
        return np.array(list(_GetPseudoAAC(protein).values()))


class ProteinFeatureAggregator:
    """Aggregator class for feature extraction.

    Description:
        This class is used to extract features from protein sequences.
    """

    def __init__(self, extractors: List[ProteinFeatureExtraction]):
        """Initialize feature aggregator."""
        self.extractors = extractors
        self.proteins_viewed = 0

    def extract_features(self, protein):
        """Extract features from protein."""
        self.proteins_viewed += 1
        extracted_features = []
        for extractor in self.extractors:
            extracted_features.append(extractor.extract_features(protein))

        # Combine extracted features into a single matrix or vector
        combined_features = np.hstack(extracted_features)

        # print count every X
        if (self.proteins_viewed % 25000) == 0:
            logging.debug(f"Number of proteins processed: {self.proteins_viewed}")
        return combined_features


class SequentialProteinFeatureAggregator:
    """Aggregator class for feature extraction of sequential data.

    Description:
        This class is used to extract features from protein sequences.
        This returns a vectors of subsequences to utilize the inherent
        sequential information in the protein sequence.
    """

    def __init__(self, extractors: List[ProteinFeatureExtraction], kmer_size=10):
        """Initialize feature aggregator."""
        self.extractors = extractors
        self.kmer_size = kmer_size

        if self.kmer_size <= 10:
            err_msg = "Sequential Aggregator requires kmersize > 10"
            raise IncorrectYamlError(err_msg)

    def extract_features(self, protein):
        """Extract the features by calling each extractor on peptide substrings."""
        sequential_features = []

        if self.kmer_size > len(protein):
            err_msg = "Sequential Aggregator requires kmersize < protein length "
            err_msg += f"the kmer size of {self.kmer_size} > {len(protein)}. "
            err_msg += f"Protein: {protein}"
            raise IncorrectYamlError(err_msg)

        for kmer_index in range(0, len(protein) - self.kmer_size):
            protein_kmer = protein[kmer_index : kmer_index + self.kmer_size]
            extracted_features = []
            for extractor in self.extractors:
                extracted_features.append(extractor.extract_features(protein_kmer))

            # Combine extracted features into a single matrix or vector
            sequential_features.append(np.hstack(extracted_features))
        return np.array(sequential_features)


if __name__ == "__main__":
    aac = AACExtractor()
    out = aac.extract_features("KLEEQDKPRADAIMALHEHKDYQPLLRAMANVPCIDVDTAKN")
    # print(aac.aminoacid2index)
    # print(out)

    dpc = DPCExtractor(parameters={"gap_size": 0})
    out = dpc.extract_features("KLEEQDKPRADAIMALHEHKDYQPLLRAMANVPCIDVDTAKN")
    # print(out)

    iso = IsoelectricExtractor()
    out = iso.extract_features("KLEEQDKPRADAIMALHEHKDYQPLLRAMANVPCIDVDTAKN")
    # print(out)

    pseudoaac = PseudoAACExtractor()
    out = pseudoaac.extract_features("KLEEQDKPRADAIMALHEHKDYQ")
    # print(out)

    atc = ATCExtractor()
    out = atc.extract_features("KLEEQDKPRADAIM")
    # print(out)

    ctd = CTDExtractor()
    out = ctd.extract_features("KLEEQDKPRADAIM")
    # print(len(out))

    # test aggregator
    aggregator = ProteinFeatureAggregator(
        extractors=[aac, dpc, iso, pseudoaac, atc, ctd]
    )
    out = aggregator.extract_features("KLEEQDKPRADAIM")
    print(len(out))

    # test aggregator
    sequential_aggregator = SequentialProteinFeatureAggregator(
        extractors=[aac, dpc, iso, pseudoaac, atc, ctd], kmer_size=11
    )
    out = sequential_aggregator.extract_features(
        "KLEEQDKPRADAIMALHEHKDYQPLLRAMANVPCIDVDTAKN"
    )
    print(len(out))
