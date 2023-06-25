<p align="center">
<a href="https://github.com/Dreycey/PhageScanner/actions/"><img alt="Actions Status" src="https://github.com/Dreycey/PhageScanner/actions/workflows/python.yml/badge.svg"></a>
<a href='https://phage-scanner.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/phage-scanner/badge/?version=latest' alt='Documentation Status' />
</a>
<a href="https://github.com/Dreycey/PhageScanner/LICENSE.txt><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

<a href="https://github.com/Dreycey/PhageScanner/blob/dreycey/main/reports/interrogate.out"><img alt="Interogate" src="https://github.com/Dreycey/PhageScanner/blob/dreycey/main/reports/interrogate.svg"></a>

</p>

![Phage Scanner Logo](figs/logo.png)

# PhageScanner

PhageScanner is a command line tool for identifying phage virion proteins (PVPs) using sequencing data as input.

## Installation

To setup the conda environment, use the following command once conda is installed locally.

-   for linux

```
conda env create -f environment_linux.yml;
```

-   for mac

```
conda env create -f environment_mac.yml;
```

## Pipeline Usage
PhageScanner is a command line tool and machine learning pipeline for automating the process of finding genes of interest. In particular, it is useful for unifying the efforts of identifying Phage Virion Proteins, and can speed up the process of finding models and using them on metagenomic data, genomes and proteins.

1. Build the database

    - Basic usage
    ```
    python phagescanner.py database -c Path/To/Config.yaml -o path/to/output/directory/ -n name_for_files_<classname>
    ```
    - Example
    ```
    python phagescanner.py database -c configs/database_multiclass.yaml -o ./benchmarking_database/ -n benchmarking -v info

    ```
2. Traing and Test ML models
    - Basic usage
    ```
    python phagescanner.py train -c Path/To/Config.yaml -o path/to/output/directory/ -n name_for_files_<classname> -v debug
    ```
    - Example
    ```
    python phagescanner.py train -c configs/training_multiclass.yaml -o training_output -n TRAIN -v debug
    ```
3. Run on metagenomic data, genomes or proteins
    - Basic usage
    ```
    python phagescanner.py predict -c Path/To/Config.yaml -o path/to/output/directory/ -t ("reads", "genome", or "protein") -n name_for_files_<classname> -i <input file> -v debug
    ```
    - Example (genomes)
    ```
    python phagescanner.py predict -c configs/prediction.yaml -t "genome" -o prediction_output -n "genomes" -i examples/GCF_000912975.1_ViralProj227117_genomic.fna -v debug
    ```
    - Example (reads)
    ```
    python phagescanner.py predict -c configs/prediction.yaml -t "reads" -o prediction_output -n "OUTPREFIX" -i examples/test_c100000_n10_e0.0.fq -v debug
    ```
    - Example (proteins)
    ```
    python phagescanner.py predict -c configs/prediction.yaml -t "protein" -o prediction_output -n OUTPREFIX -i examples/Phage_Collar_proteins.fa -v debug
    ```

# Notes

1. Database configuration names must match those in in the `DatabaseAdapterNames` Enum. If you want to create a new database adapter, make sure to add the name to the `DatabaseAdapterNames` enum. In addition, you must ensure these names match the configuration file, or the database query will be ignored.
2. Each "pipeline", as specified in the `Pipelines.py` file, is tied to a specific configuration class that is directly coupled with a particular configuration file. If you want to change the names for a particular pipeline, you must update the corresponding class. This design pattern allows for easily changing the configuration dependencies for a particular pipeline.
3. Make sure the configuration files have names that are one word - without spaces.