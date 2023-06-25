""" Database Adapter testing.

Description:
    This module contains the tests for the
    DatabaseAdapter module. It tests each class that
    implements the the DatabaseAdapter abstract interface.
"""
import unittest
from unittest.mock import patch

from requests.adapters import HTTPAdapter

from PhageScanner.main.database_adapters import EntrezAdapter


class TestEntrezAdapter(unittest.TestCase):
    """Class for testing the EntrezAdapter interface."""

    def setUp(self):
        """Create a new database adapter for Entrez."""
        self.entrez_adapter = EntrezAdapter()

    def test_get_phanns_query(self):
        """Test get_phanns_query.

        Description:
            Test if the phanns query is working as intended.
        """
        # Basic test
        query = "protein[Title]"
        result = self.entrez_adapter.get_phanns_query(query)
        expected_result = (
            "(protein[Title]) AND phage[Title] NOT hypothetical[Title] "
            "NOT putative[Title] AND 50:1000000[SLEN] NOT putitive[Title] "
            "NOT probable[Title] NOT possible[Title] NOT unknown[Title] "
        )
        self.assertEqual(result, expected_result)

        # Test with extra parameter
        query = "protein[Title]"
        extra = "extra[Title]"
        result = self.entrez_adapter.get_phanns_query(query, extra=extra)
        expected_result += extra
        self.assertEqual(result, expected_result)

    @patch.object(EntrezAdapter, "esearch")
    @patch.object(EntrezAdapter, "efetch")
    def test_query(self, mock_efetch, mock_esearch):
        """Test the query method.

        Description:
            Creates a mock EntrezAdapter fo testing if the
            query is working as intended.
        """
        # Mocking the esearch method
        mock_esearch.return_value = [["id1", "id2"], ["id3", "id4"]]

        # Mocking the efetch method
        mock_efetch.return_value = ">id1\nATCG\n>id2\nATCG\n"

        # Now call the query method
        result = list(self.entrez_adapter.query("protein"))

        # Check if esearch is called with the right parameters
        mock_esearch.assert_called_once_with("protein")

        # Check if efetch is called with the right parameters
        mock_efetch.assert_any_call(["id1", "id2"])
        mock_efetch.assert_any_call(["id3", "id4"])

        # Check if the method yields the correct output
        expected_result = [">id1\nATCG\n>id2\nATCG\n", ">id1\nATCG\n>id2\nATCG\n"]
        self.assertEqual(result, expected_result)

    def test_init(self):
        """Test the init method.

        Description:
            Tests that the init method for the Entrez adapter
            is working as expected.
        """
        # Create an instance of EntrezAdapter
        entrez_adapter = EntrezAdapter()

        # Check if an instance is successfully created
        self.assertIsInstance(entrez_adapter, EntrezAdapter)

        # Check the session object is created
        self.assertIsNotNone(entrez_adapter.session)

        # Check the session object has the correct properties
        http_adapter = entrez_adapter.session.get_adapter("https://")
        self.assertIsInstance(http_adapter, HTTPAdapter)
        self.assertEqual(http_adapter.max_retries.total, 5)
        self.assertEqual(http_adapter.max_retries.backoff_factor, 0.25)
        self.assertListEqual(
            http_adapter.max_retries.status_forcelist, [500, 502, 503, 504]
        )


if __name__ == "__main__":
    unittest.main()
