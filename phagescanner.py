"""
PhageScanner is a tool that can be used to automate the testing of various
bacteriophage virion protein (PVP) methods. It has been written such that
database retrieval/preprocessing, training/testing, and application can be 
quickly implemented and tested.                            

For help: python phagescanner.py -h
"""
import os
from enum import Enum
import argparse
import sys
import PhageScanner.main.pipelines as pipelines
from PhageScanner.main.pipelines import PipelineNames
import PhageScanner.main.utils as utils
from pathlib import Path
import logging




def parseArgs(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        help="Choose which pipeline to run", dest="sub_parser"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true")

    # if database pipeline
    database_pipeline_parser = subparsers.add_parser(
        PipelineNames.database.value, help="Run the database curation pipeline"
    )
    database_pipeline_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="path to the configuration file",
        required=True,
    )
    database_pipeline_parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Path to store the files. (will NOT overwrite existing files with the same name/path)",
        required=True,
    )
    database_pipeline_parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Name for the database files. Will be formatted as <name>_<classname>.extension",
        required=True,
    )
    database_pipeline_parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        help="Level of verbosity. Options: info, debug",
        required=False,
        default="info"
    )

    # if training pipeline
    train_pipeline_parser = subparsers.add_parser(
        PipelineNames.train.value, help="Run the training/testing pipeline"
    )
    train_pipeline_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="path to the configuration file",
        required=True,
    )
    train_pipeline_parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Path to store the files. (will NOT overwrite existing files with the same name/path)",
        required=True,
    )
    train_pipeline_parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Name for the database files. Will be formatted as <name>_<classname>.extension",
        required=True,
    )
    train_pipeline_parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        help="Level of verbosity. Options: info, debug",
        required=False,
        default="info"
    )

    # if prediction pipeline
    predict_pipeline_parser = subparsers.add_parser(
        PipelineNames.predict.value, help="Run the prediction/application pipeline"
    )
    predict_pipeline_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="path to input file.",
        required=True,
    )
    predict_pipeline_parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="Type of input file (genome, protein, or reads)",
        required=True,
    )
    predict_pipeline_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="path to the configuration file",
        required=True,
    )
    predict_pipeline_parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Path to store the files. (will NOT overwrite existing files with the same name/path)",
        required=True,
    )
    predict_pipeline_parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Name for the database files. Will be formatted as <name>_<classname>.extension",
        required=True,
    )
    predict_pipeline_parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        help="Level of verbosity. Options: info, debug",
        required=False,
        default="info"
    )

    return parser.parse_args(argv)

def main():
    def run_database_pipeline():
        print(f"Running database creation pipeline...")
        print(f"Note: This step is time consuming.")
        utils.LogUtils.configure_logging(args.out / "database_pipeline.log", args.verbosity)
        logging.info("Database creation pipeline")
        db_pipeline = pipelines.DatabasePipeline(config=args.config,
                                                 pipeline_name=args.name,
                                                 directory=args.out
                                                 )
        db_pipeline.run()

    def run_train_pipeline():
        print(f"Running training pipeline...")
        utils.LogUtils.configure_logging(args.out / "training_pipeline.log", args.verbosity)
        logging.info("Training and Testing pipeline")
        train_pipeline = pipelines.TrainingPipeline(config=args.config,
                                                 pipeline_name=args.name,
                                                 directory=args.out
                                                 )
        train_pipeline.run()

    def run_predict_pipeline():
        print(f"Running prediction pipeline...")
        utils.LogUtils.configure_logging(args.out / "prediction_pipeline.log", args.verbosity)
        logging.info("Prediction pipeline")
        train_pipeline = pipelines.PredictionPipeline(input=args.input,
                                                      input_type=args.type,
                                                      config=args.config,
                                                      pipeline_name=args.name,
                                                      directory=args.out
                                                     )
        train_pipeline.run()

    # arguments
    args = parseArgs(sys.argv[1:])

    # map subparsers to their corresponding functions and messages
    actions = {
        PipelineNames.database.value: run_database_pipeline,
        PipelineNames.train.value: run_train_pipeline,
        PipelineNames.predict.value: run_predict_pipeline,
    }

    # run benchmark type specified
    action = actions.get(args.sub_parser)
    if action:
        action()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
