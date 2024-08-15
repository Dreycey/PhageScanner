""" Database Clustering Adapters.

Description:
    This module contains the tests for the
    ClusteringAdapter module.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

from PhageScanner.main.tool_wrappers.clustering_wrappers import CDHitWrapper


class TestCDHitAdapter(unittest.TestCase):
    """Test CDHitAdapter."""

    def setUp(self):
        """Create input for testing the CDHitAdapter."""
        self.cdhit_adapter = CDHitWrapper(tool_path="", threads=4)
        self.fasta_file = Path("test.fasta")
        self.outpath = Path("out.clstr")
        self.identity = 90

    @patch("PhageScanner.main.utils.CommandLineUtils.execute_command")
    def test_cluster(self, mock_execute_command):
        """Test cluster method correctly forms and executes a command."""
        expected_command = f"{self.cdhit_adapter.tool_exe} -i {self.fasta_file} "
        expected_command += f"-o {self.outpath} -c {self.identity}"

        self.cdhit_adapter.cluster(self.fasta_file, self.outpath, self.identity)

        # Check that execute_command was called once with the expected command
        mock_execute_command.assert_called_once_with(expected_command)
