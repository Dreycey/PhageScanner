"""This pipeline interface enforces each inheriting class impliments the run method."""
import time
from abc import ABC, abstractmethod


class Pipeline(ABC):
    """Abstract class for creating new pipelines"""

    # get time of the pipeline run
    pipeline_start_time = time.ctime()

    @abstractmethod
    def run(self):
        """Run the pipeline."""
        pass
