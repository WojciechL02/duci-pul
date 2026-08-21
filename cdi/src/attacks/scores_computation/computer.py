from torch import Tensor as T
from src.attacks import DataSource


class ScoreComputer(DataSource):
    def check_data(self, data: T) -> None:
        """
        Check that the data is in the correct format
        """
        assert (
            data.shape[0] == self.total_samples
        ), "Data shape is incorrect: {}".format(data.shape)

    def run(self, data: T, *args, **kwargs) -> None:
        """
        Compute scores
        """
        # 1. Collect scores for members and nonmembers
        scores = self.process_data(data, *args, **kwargs)
        # 2. Run assertions
        self.check_data(scores)
        # 3. Save scores
        self.save(scores)
