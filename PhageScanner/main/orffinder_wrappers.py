"""This module presents wrappers for tools performing ORF finding.

Description:
    This module contains wrappers for tools performing ORF searching.
    Of note, this module currently only contains a wrapper for
    Phanotate, since it is focued on bacteriophages, but other ORF
    finding tools can be added for other species.
"""
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import re

from PhageScanner.main.exceptions import IncorrectYamlError, MissingFileError
from PhageScanner.main.utils import CommandLineUtils


class OrfFinderWrapperNames(Enum):
    """Names of orf-finding tool adapters.

    Description:
        This enum contains the names of orf-finding tool adapters.
        Of note, these names MUST match the names of the
        tool specified in the configuration file.
    """

    phanotate = "PHANOTATE"

    @classmethod
    def get_orffinding_tool(cls, name):
        """Return the the corresponding orf-finder wrapper (Factory-like pattern)"""
        name2wrapper = {
            cls.phanotate.value: PhanotateWrapper(),
        }
        wrapper = name2wrapper.get(name)

        if wrapper is None:
            tools_available = ",".join(name2wrapper.keys())
            exception_string = (
                "The Assembly tool requested in the Yaml File is not available. "
            )
            exception_string += f"The requested tool in the Yaml is: {name}. "
            exception_string += f"The options available are: {tools_available}"
            raise IncorrectYamlError(exception_string)
        return wrapper


class OrfFinderWrapper(ABC):
    """This abstract class provides an interface to assembler tools."""

    @abstractmethod
    def find_orfs(self, fasta_path: Path, outpath: Path):
        """Find orfs given a set of contigs/genomes."""
        pass


class PhanotateWrapper(OrfFinderWrapper):
    """This class defines the wrapper for phanotate."""

    def __init__(self):
        """Instantiate a phanotate wrapper for finding ORFs."""
        self.tool_exe = "phanotate.py"  # change this if installing locally.

    def find_orfs(self, fasta_path: Path, outpath: Path) -> Path:
        """Find orfs given a set of contigs/genomes.

        Returns:
            path to the output fasta ORFs.
        """
        command = f"{self.tool_exe} -f fasta -o {outpath} {fasta_path}"

        # run the command.
        logging.debug(f"Running command for phanotate: {command}")
        if not os.path.isfile(outpath):
            logging.info(f"Running Phanotate on: {outpath}")
            CommandLineUtils.execute_command(command)
        else:
            logging.info(f"Skipping finding ORFs, file exists: {outpath}")

        # make sure output exists
        if not os.path.isfile(outpath):
            error_msg = "The expected output path for Phanotate does not exist.. "
            error_msg += "This could be for many reasons, but the best way to test is "
            error_msg += "to run the pipeline again in debug mode (-v debug) and to "
            error_msg += (
                "test out the phanotate command alone to see why it's not working."
            )
            raise MissingFileError(error_msg)

        return outpath

    @staticmethod
    def get_info_from_name(fasta_entry_name: str):
        """Get information from the fasta entry name."""

        # Pattern expects Phanotate output of "NC_022762.1_CDS_[4760..5803] [note=score:-1.440691E+08]"
        pattern = r"(\w+\.\d+)_CDS_\[(\d+)\.\.(\d+)\] \[note=score:(-?\d+\.\d+E[+-]\d+)\]"
        logging.info(fasta_entry_name)
        match = re.search(pattern, fasta_entry_name)

        accession_id = match.group(1)
        start_pos = int(match.group(2))
        end_pos = int(match.group(3))
        score = float(match.group(4))

        return accession_id, start_pos, end_pos, score
