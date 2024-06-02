<p align="center">
<a href="https://github.com/Dreycey/PhageScanner/actions/"><img alt="Actions Status" src="https://github.com/Dreycey/PhageScanner/actions/workflows/testing_workflows.yml/badge.svg"></a>
<a href="https://github.com/Dreycey/PhageScanner/blob/master/LICENSE.txt"><img alt="License: GPL-3.0" src="https://img.shields.io/badge/license-GPL--3.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/Dreycey/PhageScanner/blob/dreycey/master/reports/interrogate.out"><img alt="Interogate" src="https://github.com/Dreycey/PhageScanner/blob/master/reports/interrogate.svg"></a>
</p>
 

![Phage Scanner Logo](misc/logo.png)

# PhageScanner

PhageScanner is a command line tool for identifying phage virion proteins (PVPs) using metagenomic sequencing data as input.  For comprehensive information about installation and usage, please visit the [PhageScanner Wiki](https://github.com/Dreycey/PhageScanner/wiki).

Subscribe to email list: <a href="http://eepurl.com/ivMTlY"><img alt="Subscribe" src="https://img.shields.io/badge/Subscribe-green"></a> 

Unsubscribe from email list: <a href="https://gmail.us13.list-manage.com/unsubscribe?u=d11fd2924efec07fab20ba388&id=a7720cf873"><img alt="Unsubscribe" src="https://img.shields.io/badge/Unsubscribe-red"></a>

## Installation

**NOTE**: PhageScanner is only available on 64-bit macOS and Ubuntu linux. To run the tool on windows, we recommend installing the *Ubuntu* Windows Subsystem for Linux (WSL). This limitation is due to some of the underlying tools PhageScanner uses, including: cd-hit, phanotate, and megahit. These all have C++ dependencies that are not inherently available on windows.

### Installing direct dependencies
The python dependencies can be installed using the `requirements.txt` file provided in the primary repository.
```
python -m pip install -r requirements.txt
```

### Installing command line tool dependencies
There are several command line tools that PhageScanner uses within the pipeline: (1) CD-HIT, (2) BLAST, (3) Megahit, and (4) Phanotate. Many of these tools are commonly-used bioinformatics tools that you may already have installed. However, please refer to the [PhageScanner Wiki](https://github.com/Dreycey/PhageScanner/wiki) if you'd like more guidance installing these dependencies.


## Pipeline Usage
PhageScanner is a command line tool and machine learning pipeline for automating the process of finding genes of interest. In particular, it is useful for unifying the efforts of identifying Phage Virion Proteins, and can speed up the process of finding models and using them on metagenomic data, genomes and proteins.

1. Build the database
    - Basic usage
    ```
    python phagescanner.py database -c Path/To/Config.yaml -o path/to/output/directory/ -n name_for_files_<classname>
    ```
    - Example (multiclass pvps)
    ```
    python phagescanner.py database -c configs/multiclass_pvps/database_multiclass.yaml -o ./benchmarking_database/ -n benchmarking -v info
    ```
2. Training and Test ML models
    - Basic usage
    ```
    python phagescanner.py train -c Path/To/Config.yaml -o path/to/output/directory/ -n name_for_files_<classname> -v debug
    ```
    - Example (multiclass pvps)
    ```
    python phagescanner.py train -c configs/multiclass_pvps/training_multiclass.yaml -o training_output -n TRAIN -v debug
    ```
3. Run on metagenomic data, genomes or proteins
    - Basic usage
    ```
    python phagescanner.py predict -c Path/To/Config.yaml -o path/to/output/directory/ -t ("reads", "genome", or "protein") -n name_for_files_<classname> -i <input file> -v debug
    ```
    - Example (genomes)
    ```
    python phagescanner.py predict -c configs/multiclass_pvps/prediction_multiclass.yaml  -t "genome" -o prediction_output -n "genomes" -i examples/GCF_000912975.1_ViralProj227117_genomic.fna -v debug
    ```

# PhageScanner GUI

PhageScanner has a GUI for viewing the results of the prediction pipeline to allow for scraping proteins of interest. This GUI is a visual tool for viewing the results of the prediction pipeline. The benefit of this GUI is that it allows for vissually mining proteins that may be interesting for further analysis, or for observing where the proteins appear within a genome or contig (along with synteny).

![Phage Scanner GUI](misc/gui_image.png)

## Usage

1. Create images from the output of running the `predict` pipeline.
    - run the `predict` pipeline on genomes or reads
    ```
    python phagescanner.py predict -c configs/prediction.yaml -t "genome" -o prediction_output -n "genomes" -i examples/GCF_000912975.1_ViralProj227117_genomic.fna -v debug
    ```
    - use output from the `predict` pipeline to create images
    ```
    python phagescanner_gui.py create_images -p prediction_output/genomes_predictions.csv -o output_images/
    ```
2. Open the GUI using the path to the prediction output and the images path.
    - Open the GUI
    ```
    python phagescanner_gui.py gui -p prediction_output/genomes_predictions.csv -o output_images/
    ```

# Notes

1. Database configuration names must match those in in the `DatabaseAdapterNames` Enum. If you want to create a new database adapter, make sure to add the name to the `DatabaseAdapterNames` enum. In addition, you must ensure these names match the configuration file, or the database query will be ignored.
2. Each "pipeline", as specified in the `Pipelines.py` file, is tied to a specific configuration class that is directly coupled with a particular configuration file. If you want to change the names for a particular pipeline, you must update the corresponding class. This design pattern allows for easily changing the configuration dependencies for a particular pipeline.
3. Make sure the configuration files have names that are one word - without spaces.
