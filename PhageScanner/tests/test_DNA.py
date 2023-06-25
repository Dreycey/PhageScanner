"""
Testng DNA class.

This set of unit tests tests the DNA
class for expected functionality.
"""
# Testing
import unittest

from PhageScanner.main.DNA import DNA


class TestDNA(unittest.TestCase):
    """
    Testing DNA class.

    This class contains the unit tests for testing the
    DNA class.
    """

    def test_upper(self):
        """Test DNA to protein conversion works as expected."""
        # DNA CLASS
        dna_seq = "ACGTAGCGCGCATGCGCGATCGATCGTAGCGCGCGCAGTA"

        # assertions
        self.assertEqual(DNA.dna2protein(dna_seq), "T*RACAIDRSARS")


if __name__ == "__main__":
    unittest.main()

# # PROTEIN CLASS
# prot_seq = "GTGTGTAYRACAIDRSARSACAIDRSATAYWYPTI"
# prot_obj = Protein(prot_seq, codon_table)
# prot_obj.protein2dna()
# prot_obj.chemical_sequence()
# prot_obj.secondary_structure()
# prot_obj.protein2vec()
