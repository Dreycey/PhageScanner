"""Module defines a wrapper aronud BLAST.

Description:
    This module is used to create a wrapper around BLAST.
    The goal is to be able to create a blast database
    using a training dataset, and allow for classification
    based on the training data.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PhageScanner.main.utils import CommandLineUtils
from PhageScanner.main.exceptions import IncorrectValueError



class BLASTWrapper(object):
    """A wrapper around BLAST.
    
    Description:
        This module is used to create a wrapper
        around BLAST.
    """

    def __init__(self):
        """Constructor for the BLAST wrapper."""

        self.makedbcmd = "makeblastdb"
        self.querycmd = "blastp"
        self.dbpath = None #TODO: should there be something more consistent here? if not, raise an exception.

    def create_database(self, fasta_file:Path, db_name:Path):
        """Creates a BLAST database.

        Description:
            This method creates a BLAST database using 
            numpy arrays consisting of training data and
            classifications per proteins.

        Note:
            Creates a local blast database:
                                1. DBNAME.psq
                                2. DBNAME.phr
                                3. DBNAME.pin
        """
        command = f"{self.makedbcmd} "
        command += f"-in {fasta_file} "
        command += f"-input_type fasta "
        command += f"-dbtype prot "
        command += f"-out {db_name}"

        # save blast database
        self.dbpath = db_name

        # run the command.
        CommandLineUtils.execute_command(command)

    def query(self, fasta_file: Path, outputfile: Path, threads: int=1):
        """Query the blast database."""
        # error
        if self.dbpath is None:
            raise IncorrectValueError("dbpath is required, first build the blast database.")

        # run command
        command = f"{self.querycmd} "
        command += f"-query {fasta_file} "
        command += f"-db {self.dbpath} "
        command += f"-out {outputfile} "
        command += f"-num_threads {threads} "
        command += "-max_target_seqs 1 "
        command += f'-outfmt "6 qseqid sseqid score"'

        # run the command.
        CommandLineUtils.execute_command(command)



if __name__ == "__main__":
    blast_wrapper = BLASTWrapper()

    # args needed
    if len(sys.argv) != 4:
        print("Usage: python3 blast_wrapper.py <fasta for DB> <DBpath & name> <query fasta>")
        print("python PhageScanner/main/blast_wrapper.py examples/example_proteins.fa DBNAME examples/Phage_Collar_proteins.fa")
        print("NOTE: to use this standalone, the import paths must be changed. This was used for testing.")
        exit(1)


    # create blast database.
    blast_wrapper.create_database(fasta_file=Path(sys.argv[1]), db_name=Path(sys.argv[2]))
