""" These are custom exceptions for the pipeline.

Description:
    The exceptions here are for more refined exceptions
    dealing with the exact problem occuring.
"""


class PipelineExceptionError(Exception):
    """Base exception for pipeline-related errors."""

    pass


class ProteinError(PipelineExceptionError):
    """Raised when there is an issue with a given protein."""

    pass


class IncorrectYamlError(PipelineExceptionError):
    """Raised when there is an issue with the YAML configuration."""

    pass


class MissingFileError(PipelineExceptionError):
    """Raised when a required file is missing."""

    pass


class PipelineCommandError(PipelineExceptionError):
    """Raised when a command has an error."""

    pass


class IncorrectValueError(PipelineExceptionError):
    """Raised when an incorrect value is provided."""

    pass


class ExecutableNotFoundError(PipelineExceptionError):
    """Raised when a required executable is not found."""

    pass
