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
import hashlib
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from PhageScanner.main.exceptions import (
    IncorrectValueError,
    IncorrectYamlError,
    ProteinError,
)
from PhageScanner.third_party.CTD import CalculateCTD
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
    tpc = "TPC"
    iso = "ISO"
    pseudoaac = "PSEUDOAAC"
    atc = "ATC"
    ctd = "CTD"
    protein_seq = "PROTEINSEQ"
    hash_seq = "HASH_SEQ"
    onehot = "SEQUENTIALONEHOT"
    pcp = "PCP"
    chemfeatures = "CHEMFEATURES"

    @classmethod
    def get_extractor(cls, name, parameters: Optional[Dict]):
        """Return the the corresponding feature extractor (Factory pattern)"""
        name2extractor = {
            cls.aac.value: AACExtractor,
            cls.dpc.value: DPCExtractor,
            cls.tpc.value: TPCExtractor,
            cls.iso.value: IsoelectricExtractor,
            cls.pseudoaac.value: PseudoAACExtractor,
            cls.atc.value: ATCExtractor,
            cls.ctd.value: CTDExtractor,
            cls.protein_seq.value: ProteinSequenceExtractor,
            cls.hash_seq.value: HashExtractor,
            cls.onehot.value: SequentialOneHot,
            cls.pcp.value: PCPExtractor,
            cls.chemfeatures.value: ChemFeatureExtractor
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


class TPCExtractor(ProteinFeatureExtraction):
    """Extraction method for Tripeptide composition (TPC)"""

    def __init__(self, parameters: Optional[Dict] = None):
        """Instantiate TPC extract method."""
        self.aa2index = {}

        # make map
        index = 0
        for aa1 in self.canonical_amino_acids:
            for aa2 in self.canonical_amino_acids:
                for aa3 in self.canonical_amino_acids:
                    self.aa2index[aa1 + aa2 + aa3] = index
                    index += 1

    def __eq__(self, other):
        """Set classes equal if same type."""
        if isinstance(other, TPCExtractor):
            return True
        return False

    def extract_features(self, protein: str):
        """Extract Tripeptide composition.

        Description:
            This method counts the frequency of 3 cooccuring
            *adjacent* amino acids within a given protein.
        """
        # create vector (len 8000)
        tpc_vec = np.zeros(len(self.canonical_amino_acids) ** 3)

        # count cooccuring tripeptides
        for aa_p1 in range(len(protein) - 2):
            tripeptide = protein[aa_p1 : aa_p1 + 3]
            index = self.aa2index[tripeptide]
            tpc_vec[index] += 1  # increment counter at tripeptide index

        # get frequency
        for index, count in enumerate(tpc_vec):
            tpc_vec[index] = count / len(protein)

        return tpc_vec


class IsoelectricExtractor(ProteinFeatureExtraction):
    """Extraction method for obtaining an Isoelectric point from a protein

    Description:
        Many methods have this as a standalone feature, so we added
        this here for reimplementation compatibility.
    """

    def __init__(self, parameters: Optional[Dict] = None):
        """Instantiate isoelectric extract method."""
        pass

    def extract_features(self, protein: str):
        """Obtain an isoelectric point using biopython"""
        analysis = ProteinAnalysis(protein)
        return analysis.isoelectric_point()

class PCPExtractor(ProteinFeatureExtraction):
    """Extraction method for obtaining a list of biological features.

    Description:
        This feature extraction method was introduced in
        PVP-SVM, and here we provide the extractor method.
        We make use of the biopython library to extract
        these features from the protein.

    References:
        DOI: https://doi.org/10.3389/fmicb.2018.00476
        Qoute: (v) PCP: We employed 11 representative PCP 
                attributes of amino acids for feature extraction 
                1. polar
                2. hydrophobic
                3. charged, 
                4. aliphatic
                5. aromatic, 
                6. positively charged
                7. negatively charged
                8. small/tiny 
                9. large,
                10. peptide mass
    """

    def __init__(self, parameters: Optional[Dict] = None):
        """Instantiate isoelectric extract method."""
        pass

    def is_hydrophobic(self, protein: str):
        """Test of a protein is hydrophobic."""
        # Hydrophobic amino acids
        hydrophobic_aa = ['V', 'L', 'I', 'M', 'F', 'W', 'P']

        # Count hydrophobic residues
        hydrophobic_count = sum(1 for aa in protein if aa in hydrophobic_aa)
        
        # Return True if more than half of the residues are hydrophobic
        return hydrophobic_count > len(protein) / 2

    def polarity(self, protein_sequence):
        """Returns the frequency of polar residues."""
        # polar amino acids
        polar_aa = ['S', 'T', 'N', 'Q', 'Y']

        # count polar residues
        polar_count = sum(1 for aa in protein_sequence if aa in polar_aa)

        # calculate and return the proportion of polar residues
        return polar_count / len(protein_sequence)

    def is_aliphatic(self, protein_sequence):
        """Check if a protein sequence is aliphatic"""
        # Define aliphatic amino acids
        aliphatic_aa = ['A', 'V', 'I', 'L']

        # Count aliphatic residues
        aliphatic_count = sum(1 for aa in protein_sequence if aa in aliphatic_aa)

        # Return True if more than half of the residues are aliphatic
        return aliphatic_count > len(protein_sequence) / 2

    def extract_features(self, protein: str):
        """Obtain an isoelectric point using biopython"""
        analysis = ProteinAnalysis(protein)
        # obtain properties
        is_poscharged = analysis.charge_at_pH(pH=7) > 0
        is_negcharged = analysis.charge_at_pH(pH=7) < 0
        is_small = len(protein) < 50
        is_large = len(protein) > 200
        # create feature vector
        bio_features = [
            self.polarity(protein), # 1. polarity
            self.is_hydrophobic(protein), # 2. hydrophobic
            analysis.charge_at_pH(pH=7), # 3. charged
            self.is_aliphatic(protein), # 4. aliphatic
            analysis.aromaticity(), # 5. aromatic
            is_poscharged, # 6. positively charged
            is_negcharged, # 7. negatively charged
            is_small, # 8. small/tiny
            is_large, # 9. large
            analysis.molecular_weight(), # 10. peptide mass
        ]
        return bio_features
    
class ChemFeatureExtractor(PCPExtractor):
    """Extraction method for obtaining many physical/chemical features.

    Description:
        This method build upon the PCP extractor used in PVP-SVM.
    """

    def __init__(self, parameters: Optional[Dict] = None):
        """Instantiate isoelectric extract method."""
        pass

    def extract_features(self, protein: str):
        """Obtain an isoelectric point using biopython"""
        analysis = ProteinAnalysis(protein)
        # obtain properties
        is_poscharged = analysis.charge_at_pH(pH=7) > 0
        is_negcharged = analysis.charge_at_pH(pH=7) < 0
        is_small = len(protein) < 50
        is_large = len(protein) > 200
        helix, turn, sheet = analysis.secondary_structure_fraction()
        # create feature vector
        bio_features = [
            self.polarity(protein), # 1. polarity
            self.is_hydrophobic(protein), # 2. hydrophobic
            analysis.charge_at_pH(pH=1.0), # 3. charge at acidic env
            analysis.charge_at_pH(pH=7), # 4. charge in water
            analysis.charge_at_pH(pH=10.0), # 5. charge in basic env
            self.is_aliphatic(protein), # 6. aliphatic
            analysis.aromaticity(), # 7. aromatic
            is_poscharged, # 8. positively charged
            is_negcharged, # 9. negatively charged
            is_small, # 10. small/tiny
            is_large, # 11. large
            analysis.molecular_weight(), # 12. peptide mass
            analysis.isoelectric_point(), # 13. isoelectric point
            # analysis.instability_index(), # 14. instability index
            # analysis.flexibility(), # 15. Flexibility
            # analysis.gravy(), # 16. Gravy index
            helix, # 17. Helix frequency
            turn, # 18. Turn frequency
            sheet, # 19. Betasheet frequency
            #analysis.molar_extinction_coefficient(), # 20. Molar extinction coefficient
        ]
        return bio_features


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


class SequentialOneHot(ProteinFeatureExtraction):
    """Extraction method for obtaining a 20x2000 matrix of one hot encodings.

    Description:
        The CNN and RNN methods need sequential or 2D information to
        perfrom predictions. To enable this on proteins, we take the same
        approach as Zhencheng Fang et al. (2022) and perform a one hot
        encoding for proteins less than 2000aa in length.

    Note:
        Of particular note: For proteins longer than 2000aa, we only take the
        first 2000 amino acids. Zhencheng Fang et al. (2022) completely neglected
        these proteins, but we need to allow for them as they may appear in metagenomic
        data or genomes. For proteins less than 2000aa, these are 'padded' with zeros
        at the end.

    Reference:
        Zhencheng Fang et al. (2022) - DOI: https://doi.org/10.1093/gigascience/giac076
    """

    def __init__(self, parameters: Optional[Dict] = None):
        """Instantiate tokenization extract method."""
        self.aa2index = {aa: ind for ind, aa in enumerate(self.canonical_amino_acids)}
        self.matrix_length = 1000

    def extract_features(self, protein: str):
        """Obtain an tokenization of the protein sequence."""
        tokenized_protein = np.zeros(
            (self.matrix_length, len(self.canonical_amino_acids))
        )
        for index, aa in enumerate(protein):
            if index >= self.matrix_length:
                break
            aa_index = self.aa2index[aa]
            tokenized_protein[index][aa_index] = 1
        return tokenized_protein


class HashExtractor(ProteinFeatureExtraction):
    """Extraction method implimenting a hashed vector of a protein.

    Description:
        This method was thought of as a promising
        approach to allow for larger kmer sizes while preventing
        large feature vectors. Instead of growing the feature
        vector to larger kmer sizes, we can instead hash arbitrary
        sized kmers into a fixed size feature vector. Much similiar
        to the approach used by bloom filters, this method differs as
        we incrememnt each hashed index and turn this into a frequency.
    """

    def __init__(self, parameters: Optional[Dict] = None):
        """Instantiate hash extract method."""
        # raise error if None passed to parameter
        if parameters is None:
            raise IncorrectValueError(
                "HashExtractor: must have vec_size param! Not None"
            )

        # get parameters.
        self.vec_size = parameters.get("vec_size", 50)
        self.kmer_size = parameters.get("kmer_size", 50)

    def extract_features(self, protein: str):
        """Obtain an vector hased to a given size."""
        hash_vec = np.zeros(self.vec_size)

        # ensure proteins is uft-8
        protein = protein.encode("utf-8")

        if len(protein) < self.kmer_size:
            error_message = (
                "The kmer_size is greater than the length of the protein string. "
            )
            error_message += (
                f"kmer size {self.kmer_size} > {len(protein)} protein length"
            )
            raise IncorrectYamlError(error_message)

        # get hashed kmers.
        for index in range(len(protein) - self.kmer_size + 1):
            self.hash_object = hashlib.sha256()  # create hash function
            self.hash_object.update(protein[index : index + self.kmer_size])  # add kmer
            hash_index = int(self.hash_object.hexdigest(), 16) % self.vec_size
            hash_vec[hash_index] = 1  # increment hash vec index

        # turn into frequencies.
        hash_vec /= len(protein) - self.kmer_size

        return hash_vec


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
            if type(extractor) == SequentialOneHot:
                return extracted_features

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

    def __init__(self, extractors: List[ProteinFeatureExtraction], segment_size=10):
        """Initialize feature aggregator."""
        self.extractors = extractors
        self.segment_size = segment_size

        if self.segment_size <= 1:
            err_msg = "Sequential Aggregator requires sequential > 1"
            raise IncorrectYamlError(err_msg)

    def extract_features(self, protein):
        """Extract the features by calling each extractor on peptide substrings."""
        # Ensure the protein is larger than the segment size
        if self.segment_size > len(protein):
            err_msg = "Sequential Aggregator requires kmersize < protein length "
            err_msg += f"the kmer size of {self.segment_size} > {len(protein)}. "
            err_msg += f"Protein: {protein}"
            raise IncorrectYamlError(err_msg)

        # initialize needed variables.
        sequential_features = []
        protein_sub_lengths = len(protein) // self.segment_size
        count = 0

        # for each segment, obtain the feature vector.
        for i in range(0, len(protein), protein_sub_lengths):
            if count == self.segment_size:
                break

            # get segment
            protein_subseq = protein[i : i + protein_sub_lengths]

            # get features for sub sequence
            extracted_features = []
            for extractor in self.extractors:
                extracted_features.append(extractor.extract_features(protein_subseq))

            # Combine extracted features into a single matrix or vector
            sequential_features.append(np.hstack(extracted_features))
            count += 1

        return [sequential_features]


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
        extractors=[aac, dpc, iso, pseudoaac, atc, ctd], segment_size=11
    )
    out = sequential_aggregator.extract_features(
        "KLEEQDKPRADAIMALHEHKDYQPLLRAMANVPCIDVDTAKN"
    )
    print(len(out))
