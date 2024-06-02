""" This module contains adapter for tools to assemble reads.

Description:
    This module contains wrappers around command line tools
    for assembling reads.
"""

import logging
import os
import shutil
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from PhageScanner.main.exceptions import IncorrectYamlError, MissingFileError
from PhageScanner.main.utils import CommandLineUtils


class AssemblyWrapperNames(Enum):
    """Names of assembler tool adapters.

    Description:
        This enum contains the names of assembler tool adapters.
        Of note, these names MUST match the names of the
        tool specified in the configuration file.
    """

    megahit_exe_name = "megahit"

    @classmethod
    def get_assembly_tool(cls, tool_path: Path):
        """Return the the corresponding assembly wrapper (Factory-like pattern)"""
        name2wrapper = {
            cls.megahit_exe_name.value: MegaHitWrapper,
        }
        wrapper = name2wrapper.get(tool_path.name)

        if wrapper is None:
            tools_available = ",".join(name2wrapper.keys())
            exception_string = (
                "The Assembly tool requested in the Yaml File is not available. ",
                f"The requested tool in the Yaml is: {tool_path.name}. ",
                f"The options available are: {tools_available}",
            )
            raise IncorrectYamlError(exception_string)
        return wrapper(tool_path=tool_path)


class AssemblerWrapper(ABC):
    """This abstract class provides an interface to assembler tools."""

    @abstractmethod
    def assemble(self, first: Path, second: Path = None) -> Path:
        """Assemble reads in a fasta file.

        Description:
            Should be able to take in 1 or 2 files if
            there are paired reads.

        Returns:
            Path to assembled reads.
        """
        pass


class MegaHitWrapper(AssemblerWrapper):
    """This class defines the wrapper for megahit."""

    def __init__(self, tool_path="megahit", threads=1):
        """Instantiate a megahit wrapper for clustering."""
        self.tool_exe = tool_path
        self.threads = threads

    def assemble(
        self, first: Path, out_directory: Path, second: Path = None, mem_frac=0.5
    ) -> Path:
        """Assemble reads using megahit.

        Returns:
            Path to assembled reads.
        """
        if second:  # paired ends.
            cmd = f"{self.tool_exe} "
            cmd += f"-1 {first} "
            cmd += f"-2 {second} "
            cmd += f"-o {out_directory} "
            cmd += f"-m {mem_frac} -t {self.threads}"
        else:
            cmd = f"{self.tool_exe} "
            cmd += f"-r {first} "
            cmd += f"-o {out_directory} "
            cmd += f"-m {mem_frac} -t {self.threads}"

        # final output file
        assembled_reads_path: Path = out_directory / "final.contigs.fa"

        # if final.contigs.fa not found, then delete directory
        if not os.path.isfile(assembled_reads_path) and os.path.isdir(out_directory):
            shutil.rmtree(out_directory)

        # run the command.
        logging.debug(f"Running command for megahit: {cmd}")
        CommandLineUtils.execute_command(cmd)

        if not os.path.isfile(assembled_reads_path):
            error_msg = "Could not find the `final.contigs.fa` file "
            error_msg += "expected for megahit. Check the log file for the tool. "
            error_msg += (
                "Also, try running in debug mode to get the command (-v debug). "
            )
            error_msg += f"Expected to find: '{assembled_reads_path}'"
            raise MissingFileError(error_msg)

        return assembled_reads_path
