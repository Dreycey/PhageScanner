""" Testing module for utilities.

Description:
    The testing framework for the phagescanner utils
    is contained within this module.
"""
import os
import unittest
from pathlib import Path

import yaml

from PhageScanner.main.exceptions import (
    IncorrectValueError,
    IncorrectYamlError,
    PipelineCommandError,
)
from PhageScanner.main.utils import (
    CommandLineUtils,
    ConfigUtils,
    CSVUtils,
    DatabaseConfig,
    get_filename,
)


class TestGetFilename(unittest.TestCase):
    """Test get filename."""

    def test_get_filename_from_string(self):
        """Test get_filename successfully gets filename from string path."""
        filename = get_filename("/path/to/file.txt")

        self.assertEqual(
            filename, "file", msg="The filename doesn't match the expected result."
        )

    def test_get_filename_from_path(self):
        """Test get_filename successfully gets filename from Path object."""
        filename = get_filename(Path("/path/to/file.txt"))

        self.assertEqual(
            filename, "file", msg="The filename doesn't match the expected result."
        )

    def test_get_filename_no_extension(self):
        """Test get_filename successfully gets filename without extension."""
        filename = get_filename("/path/to/file")

        self.assertEqual(
            filename, "file", msg="The filename doesn't match the expected result."
        )


class TestDatabaseConfig(unittest.TestCase):
    """Test the database configuration."""

    def setUp(self):
        """Set up the yaml file for the testing the database configuration.

        Description:
            This method creates an example database configuration file.
        """
        # Create a test YAML file
        self.test_yaml_path = Path("test.yaml")
        self.test_yaml_data = {
            "clustering": {"name": "CDHIT", "clustering-percentage": 90},
            "classes": [
                {
                    "name": "PVP",
                    "uniprot": "capsid AND cc_subcellular_location: virion",
                    "entrez": "bacteriophage[Organism]",
                },
                {
                    "name": "non-PVP",
                    "uniprot": "capsid NOT cc_subcellular_location: virion",
                    "entrez": "bacteriophage[Organism]",
                },
            ],
        }

        with open(self.test_yaml_path, "w") as f:
            yaml.dump(self.test_yaml_data, f)

        self.database_config = DatabaseConfig(self.test_yaml_path)

    def tearDown(self):
        """Clean up the test YAML file."""
        self.test_yaml_path.unlink()

    def test_get_classes(self):
        """Test get_classes method successfully reads classes."""
        classes = self.database_config.get_classes()

        self.assertEqual(
            classes,
            self.test_yaml_data["classes"],
            msg="The classes data doesn't match the expected result.",
        )

    def test_get_clustering_tool(self):
        """Test get_clustering_tool method successfully reads tool name."""
        tool_name = self.database_config.get_clustering_tool()

        self.assertEqual(
            tool_name,
            self.test_yaml_data["clustering"]["name"],
            msg="The tool name doesn't match the expected result.",
        )

    def test_get_clustering_threshold(self):
        """Test get_clustering_threshold method successfully obtains the threshold."""
        threshold = self.database_config.get_clustering_threshold()

        self.assertEqual(
            threshold,
            self.test_yaml_data["clustering"]["clustering-percentage"] / 100,
            msg="The threshold doesn't match the expected result.",
        )


class TestCommandLineUtils(unittest.TestCase):
    """Test the command line utility."""

    def test_execute_command_success(self):
        """Test execute_command successfully executes a shell command."""
        command = "echo Hello"
        expected_output = b"Hello\n"

        output = CommandLineUtils.execute_command(command)

        self.assertEqual(
            output,
            expected_output,
            msg="The command output doesn't match the expected result.",
        )

    def test_execute_command_fail(self):
        """Test execute_command handles command failure."""
        command = "nonexistent_command"

        with self.assertRaises(PipelineCommandError):
            CommandLineUtils.execute_command(command)


class TestConfigUtils(unittest.TestCase):
    """Test the config class within utils."""

    def setUp(self):
        """Set up the config test util configuration.

        Description:
            This method creates a fake yaml file path and
            fake corresponding data. It then dumps this file
            path to disk for testing.
        """
        # Create a test YAML file
        self.test_yaml_path = Path("test.yaml")
        self.test_yaml_data = {"key": "value"}

        with open(self.test_yaml_path, "w") as f:
            yaml.dump(self.test_yaml_data, f)

    def tearDown(self):
        """Cleanup the test YAML file and remove."""
        self.test_yaml_path.unlink()

    def test_open_config_success(self):
        """Test open_config method successfully reads a valid YAML file."""
        config_data = ConfigUtils.open_config(self.test_yaml_path)

        self.assertEqual(
            config_data,
            self.test_yaml_data,
            msg="The read data doesn't match the expected result.",
        )

    def test_open_config_nonexistent_file(self):
        """Test open_config method raises an exception with nonexistent file."""
        non_existent_file_path = Path("nonexistent.yaml")

        with self.assertRaises(IncorrectYamlError):
            ConfigUtils.open_config(non_existent_file_path)

    def test_open_config_invalid_file(self):
        """Test open_config method raises an exception with an invalid YAML file."""
        # Create an invalid YAML file
        invalid_yaml_path = Path("invalid.yaml")
        with open(invalid_yaml_path, "w") as f:
            f.write("invalid: [")

        with self.assertRaises(IncorrectYamlError):
            ConfigUtils.open_config(invalid_yaml_path)

        # Clean up the invalid YAML file
        invalid_yaml_path.unlink()


class TestCSVUtils(unittest.TestCase):
    """Test the CSV-parsing class within utils."""

    def setUp(self):
        """Set up the CSV-parsing class.

        Description:
            This sets up the name of of the output
            files for testing. Allows for the destructor
            to delete any files
        """
        self.outputcsv = "testoutput.csv"

    def tearDown(self):
        """Cleanup the test YAML file and remove."""
        os.remove(self.outputcsv)

    def test_appendcsv(self):
        """Performs a test on the csvappend method."""
        # create fake headers.
        headers = ["column1", "columnj", "columnk", "column4"]

        # test writing to the file.
        data = {"column1": "value", "columnj": 0.2, "columnk": 0.4}

        # write single row to the file.
        CSVUtils.appendcsv(
            data_dict=[data], fieldnames=headers, file_path=self.outputcsv
        )

        # check that there are two rows.
        with open(self.outputcsv, "r") as output:
            number_of_lines = len(output.readlines())
        self.assertEqual(number_of_lines, 2)

        # ensure data is written as expected.
        CSVUtils.appendcsv(
            data_dict=[data, data], fieldnames=headers, file_path=self.outputcsv
        )

        # ensure four rows.
        with open(self.outputcsv, "r") as output:
            number_of_lines = len(output.readlines())
        self.assertEqual(number_of_lines, 4)

    def test_incorrect_schema(self):
        """Ensure the schema must match."""
        # create data
        headers = ["column1", "columnj", "columnk", "column4"]

        # test writing to the file.
        data = {"column1": "value", "columnj": 0.2, "column_fake": 0.4}

        # test error
        with self.assertRaises(IncorrectValueError):
            CSVUtils.appendcsv(
                data_dict=[data], fieldnames=headers, file_path=self.outputcsv
            )

        # create good file
        data = {"column1": "value", "columnj": 0.2, "columnk": 0.4}
        CSVUtils.appendcsv(
            data_dict=[data], fieldnames=headers, file_path=self.outputcsv
        )

        # test schema must match the original
        headers = ["columnnew_1", "columnnew_2"]
        data = {"columnnew_1": "value", "columnnew_2": 0.4}

        # test error
        with self.assertRaises(IncorrectValueError):
            CSVUtils.appendcsv(
                data_dict=[data], fieldnames=headers, file_path=self.outputcsv
            )


if __name__ == "__main__":
    unittest.main()
