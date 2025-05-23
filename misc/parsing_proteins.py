""" Script for parsing protein names from a fasta file. """
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List
import re


class FastaNameParser:
    """ Class for parsing fasta file and retrieving information from the names. """
    
    def __init__(self, filepath: Path):
        self.name2proteinlengths: Dict[str, List[int]] = self._parse_fasta_file(filepath)

    def _parse_fasta_file(self, filepath: Path) -> Dict[str, List[int]]:
        """ Parse a fasta file and return a list of the protein lengths. """
        name2proteinlengths = defaultdict(list)
        with open(filepath, 'r') as file:
            fastalines = file.read().split('>')
            for line in fastalines:
                name, protein = line.split("\n")[0], "".join(line.split("\n")[1:])
                name2proteinlengths[name].append(len(protein))

        return name2proteinlengths

    def get_species_counts(self) -> Dict[str, int]:
        """ Get the species count from a dictionary of protein names. """
        organism2count = Counter()
        for proteinname in self.name2proteinlengths.keys():        
            match = re.search(r'OS=([^ ]+ [^ ]+)', proteinname)
            if match:
                organism = match.group(1)
                organism2count[organism] += 1

        return organism2count

    def get_protein_counts(self) -> Dict[str, int]:
        """ Get the protein name count from a dictionary of protein names. """
        proteinname2count = Counter()
        for proteinname in self.name2proteinlengths.keys():        
            match = re.search(r'\|[^ ]+ ([^OS]+) OS=', proteinname)
            if match:
                organism = match.group(1)
                proteinname2count[organism] += 1

        return proteinname2count


def get_go_names(filepath: Path):
    """Returns a set of all of the GO names (biological functions). """
    gotermCounter = Counter()
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            go_terms = line.split('\t')[2]
            if len(go_terms) < 1: continue
            for i in go_terms.split(';'):
                gotermCounter[i] += 1
    return gotermCounter

if __name__ == "__main__":
    fastafile_path = Path(sys.argv[1])
    
    go_terms = get_go_names(fastafile_path)
    print(go_terms)
    
    # parse the fasta file and extract info.
    # fastaparser = FastaNameParser(fastafile_path)
    # species_count = fastaparser.get_species_counts()
    # protein_count = fastaparser.get_protein_counts()
    
    # # printing examples.
    # print(species_count)
    # print(protein_count)
    
    