"""The database pipeline downloads all proteins of interest using Uniprot and Entrez."""

import logging
import os
from pathlib import Path

import numpy as np

from PhageScanner.main import database_adapters, utils
from PhageScanner.main.clustering_wrappers import ClusteringWrapperNames
from PhageScanner.main.database_adapters import DatabaseAdapterNames
from PhageScanner.main.pipelines.pipeline_interface import Pipeline
from PhageScanner.main.utils import CSVUtils, FastaUtils


class DatabasePipeline(Pipeline):
    """An database creation pipeline.

    Description:
        This class controls the creation, curation,
        and preprocessing steps for the protein retrieval.
    """

    def __init__(self, config: Path, pipeline_name: str, directory: Path):
        """Initialize the database pipeline."""
        logging.info("Running DatabasePipeline | Creating pipeline...")
        self.config_object = utils.DatabaseConfig(config)
        self.pipeline_name = pipeline_name
        self.directory = directory

        # create directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def get_fasta_path(self, class_name, identity=None):
        """Get the fasta path for proteins before clustering.

        Description:
            This obtains the fasta path for the fasta file.

        NOTE:
            If identity is set, this obtains fasta files that
            have been clustered at that particular identity
            threshold.
        """
        identity_str = ""
        if identity:
            identity = int(100 * identity)
            identity_str = f"_{identity}"
        path = self.directory / (
            self.pipeline_name + "_" + class_name + f"{identity_str}" + ".fasta"
        )
        return path

    def get_partition_csv_path(self, class_name):
        """Get the fasta path for proteins after clustering."""
        full = self.pipeline_name + "_" + class_name + ".csv"
        path = self.directory / full
        return path

    def get_proteins_from_db_adapters(self):
        """Get proteins from each database adapter and saves to local file.

        Description:
            This method is used to retrieved the proteins from each database.
            It works by going through each class, then iterating through each
            specified database query. The sequences are then appended to a fasta
            file.

        Note:
            This will not overwrite existing files with the name. Instead, it assumes
            that the existing file has already been created. This allows for rerunning
            the pipeline without having to start from scratch.
        """
        logging.info("Using database adapters to retrieve proteins.")

        for class_info in self.config_object.get_classes():
            class_name = class_info.get("name")
            logging.info(f"\t Retieving proteins for class: {class_name}")

            # create a file for storing the proteins.
            full_path = self.get_fasta_path(class_name)

            # if the file already exists, then go to next class.
            if os.path.isfile(full_path):
                logging.warning(
                    f"(Skip) Class already obtained: {class_name} | {full_path}"
                )
                continue

            # perform a query against each specified database adapter.
            with open(full_path, "a") as protein_class_file:
                for database_name, query in class_info.items():
                    if database_name == "name":
                        continue
                    logging.info(
                        f"\t Getting {class_name} proteins using DB: {database_name}"
                    )

                    # Get adapter with the name in the configuration file.
                    db_adapter = DatabaseAdapterNames.get_db_adapter(database_name)
                    if database_name == DatabaseAdapterNames.entrez.value:
                        query = database_adapters.EntrezAdapter.get_phanns_query(query)

                    # query the database adapter
                    logging.debug(f"\t\t Database: {database_name}, query: {query}")
                    count = 0
                    for batch in db_adapter.query(query=query):
                        if len(batch) > 0:
                            protein_class_file.write(batch)
                            count += batch.count(">")
                        else:
                            logging.warning(
                                f"Empty Batch! DB: {database_name}, query: {query}"
                            )

                    # report the count
                    logging.log(
                        logging.getLevelName("RESULT"),
                        f"COUNT DB:{database_name}, Class:{class_name}, Count:{count}",
                    )

                    # save count to csv.
                    db_count_csv = self.directory / "db_count.csv"
                    temp_db_count = {
                        "datetime": self.pipeline_start_time,
                        "database": database_name,
                        "class": class_name,
                        "class_count": count,
                    }
                    CSVUtils.appendcsv(
                        data_dict=[temp_db_count],  # input must be an array.
                        fieldnames=temp_db_count.keys(),
                        file_path=db_count_csv,
                    )

    def cluster_proteins(
        self, clustering_identity_threshold, input_identity_threshold=None
    ):
        """Cluster each class of proteins.

        Description:
            This method clusters all proteins pertaining to each
            class. Steps are:
                1. Get the clustering tool wrapper.
                2. Find all fasta files (should corresponding to each class).
                3. Cluster the proteins in each fasta file.
                4. Save information to CSV file.
        """
        clustering_tool = self.config_object.get_clustering_tool()
        clustering_adapter = ClusteringWrapperNames.get_clustering_tool(clustering_tool)

        # for each class fasta file, cluster the proteins.
        for class_info in self.config_object.get_classes():
            class_name = class_info.get("name")  # TODO: move to config_object
            logging.info(f"\t Clustering the class: {class_name}")

            # get path to proteins before and after clustering.
            input_file_path = self.get_fasta_path(
                class_name, identity=input_identity_threshold
            )
            output_file_path = self.get_fasta_path(
                class_name, identity=clustering_identity_threshold
            )

            # cluster proteins.
            clustering_adapter.cluster(
                fasta_file=input_file_path,
                outpath=output_file_path,
                identity=clustering_identity_threshold,
            )

            # save count to csv.
            db_count_csv = self.directory / "result_cluster_ouput.csv"
            temp_db_count = {
                "datetime": self.pipeline_start_time,
                "class_name": class_name,
                "clustering_threshold": clustering_identity_threshold,
                "cluster_count": FastaUtils.count_entries_in_fasta(
                    fasta_file=output_file_path
                ),
            }
            CSVUtils.appendcsv(
                data_dict=[temp_db_count],  # input must be an array.
                fieldnames=temp_db_count.keys(),
                file_path=db_count_csv,
            )

    def partition_proteins(
        self, clustering_identity_threshold, k_partitions=5, get_cluster_sizes=False
    ):
        """Partion the proteins.

        Description:
            Partitions the proteins allowing for the downstream training/testing
            pipeline to utilize k-fold cross validation during testing. This is
            done by first splitting the clustered proteins into k partitions for
            each specified class. Thereafter the the cluster members for each class
            added to each partition. The end result is a CSV file that paritions
            that have little similarity between partitions, regardless of the class.

        Returns:
            A CSV per class.
            columns:
                1. partition number
                2. protein length
                3. protein name
                4. sequence
        """
        for class_info in self.config_object.get_classes():
            class_name = class_info.get("name")  # TODO: move to config_object
            logging.info(f"\t Partitioning class {k_partitions}-fold: {class_name}")

            # get path to proteins before and after clustering.
            fasta_non_clustered = self.get_fasta_path(
                class_name, identity=self.config_object.get_deduplication_threshold()
            )
            fasta_clustered = self.get_fasta_path(
                class_name, identity=clustering_identity_threshold
            )

            # get clustering tool
            clustering_tool = self.config_object.get_clustering_tool()
            clustering_adapter = ClusteringWrapperNames.get_clustering_tool(
                clustering_tool
            )

            # get clusters as Dict
            # TODO: should done without storing all clusters into  memory.
            cluster_graph = clustering_adapter.get_clusters(fasta_clustered)

            # randomize the clusters
            randomized_clusters = list(cluster_graph.keys())
            np.random.shuffle(randomized_clusters)

            # save cluster sizes to csv
            if get_cluster_sizes:
                cluster_count_csv = self.directory / "cluster_sizes.csv"
                temp_cluster_count = {
                    "datetime": self.pipeline_start_time,
                    "class_name": class_name,
                    "cluster_count": len(cluster_graph.keys()),
                    "cluster_sizes": "\t".join(
                        [str(len(cluster)) for cluster in cluster_graph.values()]
                    ),
                }
                CSVUtils.appendcsv(
                    data_dict=[temp_cluster_count],
                    fieldnames=temp_cluster_count.keys(),
                    file_path=cluster_count_csv,
                )

            # obtain a dictionary of protein -> partition
            protein2partition = {}
            for i, cluster_id in enumerate(randomized_clusters):
                cluster_partition = (i % k_partitions) + 1
                # assign clusters to the same partition
                for protein_accesion in cluster_graph[cluster_id]:
                    protein2partition[protein_accesion] = cluster_partition

            # delete graph to save some space
            del randomized_clusters
            del cluster_graph

            # get accesion ids for each clusters reference/centroid protein.
            output_file = self.get_partition_csv_path(class_name)
            with open(output_file, "w") as output_csv:
                output_csv.write("partition,accession,protein,protein_length\n")
                for accession, protein in FastaUtils.get_proteins(fasta_non_clustered):
                    if (
                        accession[:19] in protein2partition
                    ):  # NOTE: TODO: CDHIT cuts names at 19 ch
                        partition = protein2partition[accession[:19]]
                        output_csv.write(
                            f"{partition},{accession},{protein},{len(protein)}\n"
                        )
                    else:
                        logging.warning(
                            f"protein {accession} was not found in clusters"
                        )

    def run(self):
        """Run the pipeline.

        Description:
            This method runs the pipeline for
            creating a new set of unique proteins
            using different input database adapters.
        """
        # config variables
        deduplication_threshold = self.config_object.get_deduplication_threshold()

        # Step 1: get proteins using database adapters.
        logging.info("Step 1 - Obtaining proteins from APIs...")
        self.get_proteins_from_db_adapters()
        logging.info("Step 1 (Finished) - Obtaining proteins from APIs...")

        # Step 2: cluster proteins to remove duplicates.
        logging.info("Step 2 - Removing duplicates...")
        self.cluster_proteins(clustering_identity_threshold=deduplication_threshold)
        logging.info("Step 2 (Finished) - Removing duplicates...")

        # Step 3: cluster proteins at the predifined clustering threshold.
        logging.info("Step 3 - Cluster the proteins...")
        self.cluster_proteins(
            clustering_identity_threshold=self.config_object.get_clustering_threshold(),
            input_identity_threshold=deduplication_threshold,
        )
        logging.info("Step 3 (Finished) - Cluster the proteins...")

        # Step 4: create k-fold partitioned clusters.
        logging.info("Step 4 - Create k-fold partitions...")
        self.partition_proteins(
            clustering_identity_threshold=self.config_object.get_clustering_threshold(),
            k_partitions=self.config_object.get_k_partition_count(),
        )
        logging.info("Step 4 (Finished) - Create k-fold partitions...")
