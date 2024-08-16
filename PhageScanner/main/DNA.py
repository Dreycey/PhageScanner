""" Methods for working with biological strings.

Description:
    This module contains classes and methods for working
    with strings representing biological entities. This
    includes DNA and Protein sequences.
"""

from dataclasses import dataclass
from typing import List

DNA_Codons = {
    # 'M' - START, '_' - STOP
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TGT": "C",
    "TGC": "C",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TTT": "F",
    "TTC": "F",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
    "CAT": "H",
    "CAC": "H",
    "ATA": "I",
    "ATT": "I",
    "ATC": "I",
    "AAA": "K",
    "AAG": "K",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATG": "M",
    "AAT": "N",
    "AAC": "N",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGA": "R",
    "AGG": "R",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "AGT": "S",
    "AGC": "S",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TGG": "W",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGA": "*",
}


@dataclass
class CodonTable:
    """Dataclass describing a codon table."""

    codon_map: dict

    def amino2codon(self, amino) -> List:
        """Map from an amino acid to possible codons."""
        possible_codons = []
        for codon, amino_i in self.codon_map.items():
            if amino == amino_i:
                possible_codons.append(codon)
        return possible_codons

    def codon2amino(self, codon):
        """Turn a codon into a amino acid."""
        return self.codon_map[codon]


class DNA:
    """This class works with DNA sequences."""

    codon_table = CodonTable(DNA_Codons)

    @classmethod
    def dna2protein(cls, dna_seq=None):
        """Turn DNA sequence into protein sequence."""
        protein_seq = ""
        for nuc_ind in range(0, len(dna_seq), 3):
            codon = dna_seq[nuc_ind : nuc_ind + 3]
            if len(codon) == 3:
                protein_seq += cls.codon_table.codon2amino(codon)
        return protein_seq
