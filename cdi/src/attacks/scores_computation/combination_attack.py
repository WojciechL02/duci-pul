from src.attacks import ScoreComputer
from torch import Tensor as T
import torch


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.attacks.utils import MIDataset, get_datasets_clf

from typing import Tuple

classifiers = {
    "lr": LogisticRegression,
}


class CombinationAttackComputer(ScoreComputer):
    def load_both_parts(self) -> Tuple[T, T]:
        filename_original = self.path_in.replace(".npz", "")
        if self.dataset_cfg.train_name in filename_original:
            filename_members = filename_original
            filename_nonmembers = filename_original.replace(
                self.dataset_cfg.train_name, self.dataset_cfg.test_name
            )
        else:
            filename_nonmembers = filename_original
            filename_members = filename_original.replace(
                self.dataset_cfg.test_name, self.dataset_cfg.train_name
            )

        members, _ = self.load("", filename_members)
        nonmembers, _ = self.load("", filename_nonmembers)

        B = members.size(0)
        return members.view(B, -1), nonmembers.view(B, -1)  # (B, F), (B, F)

    def fit_clf(self, train_dataset: MIDataset, seed: int):
        clf = classifiers[self.attack_cfg.clf](
            random_state=seed, **self.attack_cfg.kwargs
        )
        clf.fit(train_dataset.data, train_dataset.label)
        return clf

    def compute_score(self, data: T, clf) -> T:
        return torch.from_numpy(clf.predict_proba(data)[:, 1])

    def process_data(self, data: T) -> Tuple[T, T]:
        members, nonmembers = self.load_both_parts()
        train_dataset, valid_dataset, test_dataset = get_datasets_clf(
            members.reshape(members.shape[0], -1),
            nonmembers.reshape(members.shape[0], -1),
            self.config.train_samples,
            self.config.valid_samples,
            self.config.n_samples_eval,
        )
        ss = StandardScaler()

        # For the final evaluation we use all left-out samples as training samples

        train_dataset.data = torch.concat([train_dataset.data, valid_dataset.data])
        train_dataset.label = torch.concat([train_dataset.label, valid_dataset.label])

        train_dataset.data = torch.from_numpy(ss.fit_transform(train_dataset.data))
        test_dataset.data = torch.from_numpy(ss.transform(test_dataset.data))

        clf = self.fit_clf(train_dataset, self.config.seed)
        members_scores, nonmembers_scores = [], []

        for dataset in [test_dataset, train_dataset]:
            scores = self.compute_score(dataset.data, clf)

            members_scores.append(scores[: len(dataset) // 2])
            nonmembers_scores.append(scores[len(dataset) // 2 :])

        print(
            "train:",
            clf.score(train_dataset.data, train_dataset.label),
            "eval:",
            clf.score(test_dataset.data, test_dataset.label),
        )

        print(torch.cat(members_scores).shape, torch.cat(nonmembers_scores).shape)

        filename_original = self.path_in.replace(".npz", "")
        if self.dataset_cfg.train_name in filename_original:
            return torch.cat(members_scores)
        return torch.cat(nonmembers_scores)
