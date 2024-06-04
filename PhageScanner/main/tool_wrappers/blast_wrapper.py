"""Module defines a wrapper aronud BLAST.

Description:
    This module is used to create a wrapper around BLAST.
    The goal is to be able to create a blast database
    using a training dataset, and allow for classification
    based on the training data.
"""

from pathlib import Path

from PhageScanner.main.exceptions import IncorrectValueError
from PhageScanner.main.utils import CommandLineUtils


class BLASTWrapper(object):
    """A wrapper around BLAST.

    Description:
        This module is used to create a wrapper
        around BLAST.
    """

    def __init__(self, makeblastdb_exe_path="makeblastdb", blastp_exe_path="blastp"):
        """Construct for the BLAST wrapper."""
        self.makeblastdb_exe_path = makeblastdb_exe_path
        self.blastp_exe_path = blastp_exe_path
        self.dbpath = None

    def create_database(self, fasta_file: Path, db_name: Path):
        """Create a BLAST database.

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
        command = f"{self.makeblastdb_exe_path} "
        command += f"-in {fasta_file} "
        command += "-input_type fasta "
        command += "-dbtype prot "
        command += f"-out {db_name}"

        # save blast database
        self.dbpath = db_name

        # run the command.
        CommandLineUtils.execute_command(command)

    def query(self, fasta_file: Path, outputfile: Path, threads: int = 1):
        """Query the blast database."""
        # error
        if self.dbpath is None:
            raise IncorrectValueError(
                "dbpath is required, first build the blast database."
            )

        # run command
        command = f"{self.blastp_exe_path} "
        command += f"-query {fasta_file} "
        command += f"-db {self.dbpath} "
        command += f"-out {outputfile} "
        command += f"-num_threads {threads} "
        command += "-max_target_seqs 1 "
        command += '-outfmt "6 qseqid sseqid score"'

        # run the command.
        CommandLineUtils.execute_command(command)
