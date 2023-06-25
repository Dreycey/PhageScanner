""" This module contains adapter for tools to cluster proteins.

Description:
    This module contains wrappers around command line tools
    for performing clustering on proteins in a fasta file.
"""
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List

from PhageScanner.main.exceptions import IncorrectYamlError
from PhageScanner.main.utils import CommandLineUtils


class ClusteringWrapperNames(Enum):
    """Names of clustering tool adapters.

    Description:
        This enum contains the names of clustering tool adapters.
        Of note, these names MUST match the names of the
        tool specified in the configuration file.
    """

    cdhit = "cdhit"

    @classmethod
    def get_clustering_tool(cls, name):
        """Return the the corresponding cluster adapter (Factory-like pattern)"""
        name2wrapper = {
            cls.cdhit.value: CDHitWrapper(),
        }
        wrapper = name2wrapper.get(name)

        if wrapper is None:
            tools_available = ",".join(name2wrapper.keys())
            exception_string = (
                "The Clustering tool requested in the Yaml File is not available. "
            )
            exception_string += f"The requested tool in the Yaml is: {name}. "
            exception_string += f"The options available are: {tools_available}"
            raise IncorrectYamlError(exception_string)
        return wrapper


class ClusterWrapper(ABC):
    """This abstract class provides an interface to clustering tools."""

    @abstractmethod
    def cluster(self, fasta_file: Path, identity=int):
        """Cluster a fasta file of proteins using a given identity threshold."""
        pass

    @abstractmethod
    def get_clusters(self, file_prefix: Path) -> Dict[str, str]:
        """Return a dictionary of clusters."""


class CDHitWrapper(ClusterWrapper):
    """This conctrete class provides an interface to clustering using CDHit."""

    def __init__(self, threads=4):
        """Instantiate a CDHit adapter for clustering."""
        self.threads = threads
        self.tool_exe = "cd-hit"  # change this if executable is not env variable.

    def cluster(self, fasta_file: Path, outpath: Path, identity=int):
        """Cluster a fasta file of proteins using CDHit."""
        logging.debug(f"Clustering proteins for {fasta_file}")
        command = f"{self.tool_exe} -i {fasta_file} -o {outpath} -c {identity}"

        # run the command.
        CommandLineUtils.execute_command(command)

    def get_clusters(self, file_prefix: Path) -> Dict[str, List[str]]:
        """Obtain a dictionary of clusters."""
        file_path = file_prefix.with_suffix(".clstr")
        cluster = {}
        # obtain clusters
        cluster_members, cluster_name = [], "(Error) initialized cluster name"
        with open(file_path, "r") as cluster_file:
            file_read = cluster_file.readlines()
            for index, line in enumerate(file_read):
                if line.startswith(">"):
                    if index > 1:
                        cluster[cluster_name] = cluster_members
                    cluster_members = []
                    cluster_name = line.split(" ")[1].strip("\n")
                else:
                    accession = line.split(">")[1].split("...")[0].strip(" ")
                    cluster_members.append(accession)
        return cluster
