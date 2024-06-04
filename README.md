<p align="center">
<a href="https://github.com/Dreycey/PhageScanner/actions/"><img alt="Actions Status" src="https://github.com/Dreycey/PhageScanner/actions/workflows/testing_workflows.yml/badge.svg"></a>
<a href="https://github.com/Dreycey/PhageScanner/blob/master/LICENSE.txt"><img alt="License: GPL-3.0" src="https://img.shields.io/badge/license-GPL--3.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/Dreycey/PhageScanner/blob/dreycey/master/reports/interrogate.out"><img alt="Interogate" src="https://github.com/Dreycey/PhageScanner/blob/master/reports/interrogate.svg"></a>
<a href="https://hub.docker.com/r/dreyceyalbin/phagescanner"><img alt="DockerHub" src=" https://img.shields.io/docker/v/dreyceyalbin/phagescanner"></a>
</p>

![Phage Scanner Logo](misc/logo.png)

# PhageScanner

PhageScanner is a command line tool for identifying phage virion proteins (PVPs) using metagenomic sequencing data as input.  For comprehensive information about installation and usage, please visit the [PhageScanner Wiki](https://github.com/Dreycey/PhageScanner/wiki).

Subscribe to email list: <a href="http://eepurl.com/ivMTlY"><img alt="Subscribe" src="https://img.shields.io/badge/Subscribe-green"></a> 

Unsubscribe from email list: <a href="https://gmail.us13.list-manage.com/unsubscribe?u=d11fd2924efec07fab20ba388&id=a7720cf873"><img alt="Unsubscribe" src="https://img.shields.io/badge/Unsubscribe-red"></a>

## Installation (Mac and Linux)

### Installing direct dependencies
The python dependencies can be installed using the `requirements.txt` file provided in the primary repository.
```
python -m pip install -r requirements.txt
```

### Installing command line tool dependencies
There are several command line tools that PhageScanner uses within the pipeline: (1) CD-HIT, (2) BLAST, (3) Megahit, and (4) Phanotate. Many of these tools are commonly-used bioinformatics tools that you may already have installed. However, please refer to the [PhageScanner Wiki](https://github.com/Dreycey/PhageScanner/wiki) if you'd like more guidance installing these dependencies.

## Installing using Docker (Windows, Mac and Linux)
The easiest approach to using PhageScanner is to use Docker. Docker allows for PhageScanner to be usable on Windows and removes the need to install the command line tool dependencies. Follow the directions to [install docker](https://docs.docker.com/desktop/install/). For Windows, we used WSL2 to install docker (instead of Hyper-V), but both should work as intended.

### Using the Docker image host on DockerHub
PhageScanner is host on DockerHub at [https://hub.docker.com/r/dreyceyalbin/phagescanner](https://hub.docker.com/r/dreyceyalbin/phagescanner). This allows for easily downloading the Docker image and running the tool after installing Docker.

* Pull down the docker image from DockerHub
```
docker pull dreyceyalbin/phagescanner
```

* Test that the help message prints
```
docker run --rm dreyceyalbin/phagescanner --help
```

### Building Docker image locally
The docker image can be built locally to allow for more flexiblity. There are two steps involved in this process:

* Navigate to the `Docker/` directory and run:
```
docker build -t dreyceyalbin/phagescanner .
```

* Test that the help message prints
```
docker run --rm dreyceyalbin/phagescanner --help
```

## Pipeline Usage
There are three fundamental pipelines in the PhageScanner tool. Each of these pipelines feeds into the next: (1) Download the training dataset, (2) Training the machine learning models, (3) Using the models to annotate genomes and metagenomics datasets. Each pipelines is configurable to allow end-users extreme flexibility in creating new models to predict new variations of protein classes (ex. "Toxic Protein", "Phage Virion Protein", "Lysogenic"). Each example list below **should be ran from the root directory** if running the commands "as-is".

1. Build the database
    - Basic usage
    ```
    python phagescanner.py database [-h] -c CONFIG -o OUT [--cdhit_path CDHIT_PATH (Default: 'cdihit')] [-v VERBOSITY]
    ```
    - Example (multiclass pvps)
    ```
    python phagescanner.py database -c configs/multiclass_config.yaml -o ./multiclass_database/ -v info
    ```
    - Example using Docker (multiclass pvps)
    ```
    docker run --rm \
        -v "$(pwd)/configs:/app/configs" \
        -v "$(pwd)/multiclass_database:/app/multiclass_database" \
        dreyceyalbin/phagescanner database -c /app/configs/multiclass_config.yaml -o /app/multiclass_database/ -v info
    ```
2. Training and Test ML models
    - Basic usage
    ```
    python phagescanner.py train [-h] -c CONFIG -o OUT -db DATABASE_CSV_PATH [-v VERBOSITY]
    ```
    - Example (multiclass pvps)
    ```
    python phagescanner.py train -c configs/multiclass_config.yaml -o training_output --database_csv_path ./multiclass_database/ -v debug
    ```
    - Example using Docker (multiclass pvps)
    ```
    docker run --rm \
        -v "$(pwd)/configs:/app/configs" \
        -v "$(pwd)/multiclass_database:/app/multiclass_database" \
        -v "$(pwd)/training_output:/app/training_output" \
        dreyceyalbin/phagescanner train -c /app/configs/multiclass_config.yaml -o /app/training_output --database_csv_path /app/multiclass_database/ -v debug
    ```
3. Run on metagenomic data, genomes or proteins
    - Basic usage
    ```
    python phagescanner.py predict [-h] -i INPUT -t TYPE ("reads", "genome", or "protein") -c CONFIG -o training_output -n NAME -tdir TRAINING_OUTPUT
                                [--megahit_path MEGAHIT_PATH (Default: 'megahit')] [--phanotate_path PHANOTATE_PATH (Default: 'phanotate.py')]
                                [--probability_threshold PROBABILITY_THRESHOLD] [-v VERBOSITY]
    ```
    - Example (genomes; though sequencing reads and proteins can be used as input)
    ```
    python phagescanner.py predict -t "genomes" -c configs/multiclass_config.yaml -n "OUTPREFIX" -tdir .\training_output\ -o prediction_output -i examples/GCF_000912975.1_ViralProj227117_genomic.fna -v debug
    ```
    - Example using Docker (genomes)
    ```
    docker run --rm \
        -v "$(pwd)/configs:/app/configs" \
        -v "$(pwd)/examples:/app/examples" \
        -v "$(pwd)/prediction_output:/app/prediction_output" \
        -v "$(pwd)/training_output:/app/training_output" \
        dreyceyalbin/phagescanner predict -t "genome" -c /app/configs/multiclass_config.yaml -o /app/prediction_output -n "OUTPREFIX" -tdir .\training_output\ -i /app/examples/GCF_000912975.1_ViralProj227117_genomic.fna -v debug
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