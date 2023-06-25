""" Contains database adapters.

Description:
    This module contains the database adapters
    used for connecting to the database APIs.

Patterns:
    Strategy pattern is used to give all databases
    a query method that returns the same type.
"""
import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry


class DatabaseAdapterNames(Enum):
    """Names of database adapters.

    Description:
        This enum contains the names of database adapters.
        Of note, these names MUST match the names of the
        databases specified in the configuration file.
    """

    uniprot = "uniprot"
    entrez = "entrez"

    @classmethod
    def get_db_adapter(cls, name):
        """Return the the corresponding database adapter (Factory pattern)"""
        name2adapter = {
            cls.uniprot.value: UniprotAdapter(),
            cls.entrez.value: EntrezAdapter(),
        }
        return name2adapter.get(name)


class DatabaseAdapter(ABC):
    """Provides a database adapter interface."""

    @abstractmethod
    def query(self, query):
        """Query a database."""
        pass


class MockAdapter(DatabaseAdapter):
    """MockAdapter class.

    Description:
        Mock adapter for testing
    """

    def query(self, query):
        """Perfoms a mock query.

        Description:
            This method resembles the output expected
            by other adapters.
        """
        query_output_1 = """
            >example_1
            MAKLNQVTLSKIGKNGDQTLTLTPRGVNPTNGVASLSEAGAVPALEKRVTVSVAQPSRNR
            KNFKVQIKLQNPTACTRDACDPSVTRSAFADVTLSFTSYSTDEERALIRTELAALLADPL
            IVDAIDNLNPAYWAALLVASSGGGDNPSDPDVPVVPDVKPPDGTGRYKCPFACYRLGSIY
            EVGKEGSPDIYERGDEVSVTFDYALEDFLGNTNWRNWDQRLSDYDIANRRRCRGNGYIDL
            DATAMQSDDFVLSGRYGVRKVKFPGAFGSIKYLLNIQGDAWLDLSEVTAYRSYGMVIGFW
            TDSKSPQLPTDFTQFNSANCPVQTVIIIPSL
            >example_2
            MSKKAVPPIVKAQYELYNRKLNRAIKVSGSQKKLDASFVGFSESSNPETGKPHADMSMSA
            KVKRVNTWLKNFDREYWDNQFASKPIPRPAKQVLKGSSSKSQQRDEGEVVFTRKDSQKSV
            RTVSYWVCTPEKSMKPLKYKEDENVVEVTFNDLAAQKAGDKLVSILLEINVVGGAVDDKG
            RVAVLEKDAAVTVDYLLGSPYEAINLVSGLNKINFRSMTDVVDSIPSLLNERKVCVFQND
            DSSSFYIRKWANFLQEVSAVLPVGTGKSSTIVLT
            """
        query_output_2 = """
            >example_3
            MANRRQSRRRGNKNRNSATVRRAPPNRATQSSSGKVKFVKWIAASPTKLIPHIGENETSY
            GVLFDITGTTFPELSSLMSRHSRYRVLSLGARIVPYDPNCLGAHSVKVFAESVYDSSATP
            TVPSVHYLQSNGCRVVPANKQLSSPPSSDVNKYYEICSDDAVIGRIMYAWNGPGLSTARV
            GFCSFEVYADLEFDGIRE
            """

        for batch in [query_output_1, query_output_2]:
            yield batch.replace(" ", "")


class UniprotAdapter(DatabaseAdapter):
    """This class is an adapter for uniprot.

    Description:
        This class is an adapter that connects to the
        Uniprot API and retrieves protiens based on
        input queries.

    Reference:
        https://www.uniprot.org/help/api_queries
    """

    def __init__(self):
        """Instantiate Uniprot adapter."""
        self.url = "https://rest.uniprot.org/uniprotkb/search"
        # set up request handlers
        retries = Retry(
            total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504]
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def get_next_link(self, headers):
        """Obtain the next link from the header.

        Description:
            Retrieves the next link for the given headers
            to allow for batch processing.
        """
        re_next_link = re.compile(r'<(.+)>; rel="next"')
        if "Link" in headers:
            match = re_next_link.match(headers["Link"])
            if match:
                return match.group(1)

    def get_batch(self, parameters):
        """Get the next batch.

        Description:
            Given a Uniprot dictionary of uniprot API parameters,
            this method retrieves the proteins in batches.

        Args:
            parameters (dictionary): A dictionary of uniprot API parameters

        Returns: [Yields]
            Yields a batch a string of uniprot API output.
        """
        response = False
        batch_url = self.url
        while batch_url:
            # retrieve response from uniprot.
            if response:
                response = self.session.get(batch_url)
            else:
                response = self.session.get(batch_url, params=parameters)
            total = response.headers["x-total-results"]

            # check output and handle errors.
            logging.debug("Response url for uniprot get batch: {response.url}")
            response.raise_for_status()
            yield response, total
            batch_url = self.get_next_link(response.headers)

    def query(self, query):
        """Query uniprot.

        Description:
            Sends a query to Uniprot and returns a fasta string
            for the given query.

        Args:
            query (str): A Uniprot query string.

        Returns:
            A string fasta representation of the uniprot query.
        """
        params = {
            "fields": "accession,sequence",  # go_f,cc_interaction,cc_function,
            "format": "fasta",
            "query": query,
            "size": 400,
            "compressed": False,
        }
        for batch, _ in self.get_batch(params):
            if len(batch.text.splitlines()[1:]) == 0:
                logging.warning("batch is empty..")
            yield batch.text


class EntrezAdapter(DatabaseAdapter):
    """Entrez adapter class.

    Description:
        This class implements an interface to interact with the
        Entrez database.

    More information:
        https://www.ncbi.nlm.nih.gov/books/NBK25497/
    """

    def __init__(self):
        """Instantiate Entrez adapter."""
        # static Entrez information
        self.url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.db = "protein"

        # set up request handlers
        retries = Retry(
            total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504]
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    @staticmethod
    def get_phanns_query(query, extra=""):
        """Format a string for phanns.

        Description:
            modifies the search to match PhANNs
        """
        modified_query = f"({query}) AND phage[Title] NOT hypothetical[Title] "
        modified_query += (
            "NOT putative[Title] AND 50:1000000[SLEN] NOT putitive[Title] "
        )
        modified_query += "NOT probable[Title] NOT possible[Title] NOT unknown[Title] "
        modified_query += extra
        return modified_query

    def esearch(self, query, batch_size=10000) -> List[str]:
        """Return a list of URIs.

        Description:
            Uses the esearch server side function of the Entrez
            API to search for proteins in the database mathcing
            the query parameters.

        Returns:
            An HTML list of Entrez UIDs associated with the input query.
            info for queries: https://www.ncbi.nlm.nih.gov/books/NBK3837/
        """
        # Set the rettype to XML and the retmode to text
        rettype = "xml"
        retmode = "text"

        # initial query to get dataset size.
        search_url = (
            f"{self.url}esearch.fcgi?db={self.db}&term={query}&rettype={rettype}"
        )
        search_url += f"&retmode={retmode}&retmax=1"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.content, features="xml")
        dataset_size = int(soup.find("Count").text)

        # retrieve full set.
        for ret_start in range(1, dataset_size, batch_size):
            # set up the search query
            logging.debug(f"current ret_start: {ret_start}")
            search_url = (
                f"{self.url}esearch.fcgi?db={self.db}&term={query}&rettype={rettype}"
            )
            search_url += f"usehistory=y&retmode={retmode}&retstart={ret_start}"
            search_url += f"&retmax={batch_size}"
            response = requests.get(search_url)

            # Parse out xml responses
            soup = BeautifulSoup(response.content, features="xml")
            id_list = soup.find_all("Id")
            logging.debug(f" (EntrezAdapter) Retreived id count: {len(id_list)}")

            yield id_list

    def efetch(self, id_list, batch_size=200):
        """Get URIs to fetch information.

        Description:
            Uses the efetch server side function of the Entrez
            to fetch proteins based on a list of UIDs.
        """
        # initialize for the efetch server side function.
        parameters = {
            "db": {self.db},
            "rettype": "fasta",
            "retmode": "text",
            "retstart": 1,
            "retmax": {batch_size},
        }
        # retrieve each batch
        ids = [id.text for id in id_list]
        id_count = len(ids)
        fasta_string = ""
        for retstart in range(1, id_count, batch_size):
            parameters["id"] = ",".join(ids[retstart : retstart + batch_size])
            response = requests.get(f"{self.url}efetch.fcgi", params=parameters)
            fasta_string += response.content.decode("ascii")
        return fasta_string

    def query(self, query):
        """Perfom a query against Entrez.

        Description:
            Queries the Entrez database API and returns
            a string representation of the fasta output
            resulting from an input query specified for Entrez.

        Args:
            query: string

        Returns: [Yields: string]
            Fasta (string)

        References: https://www.ncbi.nlm.nih.gov/books/NBK3837/
        """
        for id_set in self.esearch(query):
            fasta_output = self.efetch(id_set)
            yield fasta_output.replace("\n\n", "\n")


if __name__ == "__main__":
    # Mock Example.
    mockdb = MockAdapter()
    fake_query = "anything"
    for batch in mockdb.query(query=fake_query):
        print(batch)

    # Uniprot Example
    uniprotdb = UniprotAdapter()
    positive_pvps = "capsid AND cc_subcellular_location: virion AND reviewed: true"
    negative_pvps = "capsid NOT cc_subcellular_location: virion AND reviewed: true"
    for batch in uniprotdb.query(query=positive_pvps):
        print(batch)

    # Entrez Example
    entrez = EntrezAdapter()  # "bacteriophage[Organism] AND "
    modified_query = entrez.get_phanns_query("portal")
    # modified_query = "bacteriophage[Organism] AND lysin"
    for batch in entrez.query(query=modified_query):
        print(batch)


# if False:  # these work.
#     url = "https://www.ebi.ac.uk/proteins/api/proteins"
#     params = {
#         "offset": "0",
#         "size": "10000",
#         "reviewed": "true",
#         "organism": "Bacteriophage",
#         "format": "fasta"
#     }

#     response = requests.get(url, params=params)

#     if response.ok:
#         filename = "bacteriophage.fasta"
#         with open(filename, "w") as f:
#             f.write(response.text)
#         print(f"FASTA sequences written to {filename}")
#     else:
#         print(
#             f"Error downloading sequences: {response.status_code}
# {response.reason}")

#     params = {
#         "offset": "0",
#         "size": "100",
#         "reviewed": "true",
#         "format": "csv",
#         "organism": "Bacteriophage",
#         "columns": "id,entry name,protein names,genes,length,organism"
#     }

#     response = requests.get(url, params=params)

#     if response.ok:
#         filename = "bacteriophage.txt"
#         with open(filename, "w") as f:
#             f.write(response.text)
#         print(f"Protein information written to {filename}")
#     else:
#         print(
#             f"Error downloading protein information:
#            {response.status_code} {response.reason}")
# if false:
#     import requests


# # import requests
# # import mysql.connector

# # class MySQLAdapter:
# #     def __init__(self, db_config):
# #         self.db_config = db_config

# #     def insert_protein(self, protein):
# #         # Connect to MySQL database
# #         cnx = mysql.connector.connect(**self.db_config)
# #         cursor = cnx.cursor()

# #         # Insert protein into database
# #         add_protein = ("INSERT INTO proteins "
# #                        "(id, name, description, sequence) "
# #                        "VALUES (%s, %s, %s, %s)")
# #         data_protein = (protein.id, protein.name, protein.description,
# protein.sequence)
# #         cursor.execute(add_protein, data_protein)

# #         # Commit changes and close connection
# #         cnx.commit()
# #         cursor.close()
# #         cnx.close()


# if False:
#     import mysql.connector

#     class MySQLAdapter:
#         def __init__(self, db_config):
#             self.db_config = db_config
#             self._connect()

#         def _connect(self):
#             self.cnx = mysql.connector.connect(**self.db_config)
#             self.cursor = self.cnx.cursor()

#         def store_protein(self, protein):
#             # Insert protein into database
#             add_protein = ("INSERT INTO proteins "
#                            "(id, name, description, sequence) "
#                            "VALUES (%s, %s, %s, %s)")
#             data_protein = (protein.id, protein.name,
#                             protein.description, protein.sequence)
#             self.cursor.execute(add_protein, data_protein)
#             self.cnx.commit()

#         def check_protein(self, protein_id):
#             # Retrieve protein from database
#             query = ("SELECT id, name, description, sequence "
#                      "FROM proteins "
#                      "WHERE id = %s")
#             data_protein = (protein_id,)
#             self.cursor.execute(query, data_protein)
#             row = self.cursor.fetchone()

#             # Check if protein is in database
#             if row is None:
#                 return False

#             # Check if any fields are missing
#             if None in row:
#                 return True

#             return False

#         def close(self):
#             self.cursor.close()
#             self.cnx.close()
