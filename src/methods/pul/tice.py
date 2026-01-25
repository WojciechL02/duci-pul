import math
import heapq
import numpy as np
from bitarray import bitarray
from typing import Dict


class TICE:
    def __init__(
        self,
        max_bepp: int = 5,
        delta: float = None,
        nb_its: int = 3,
        max_splits: int = 500,
        use_most_promising: bool = False,
        min_t: int = 10,
        n_splits: int = 3,
    ) -> None:
        self.k = max_bepp
        self.delta = delta
        self.nbIterations = nb_its
        self.maxSplits = max_splits
        self.useMostPromisingOnly = use_most_promising
        self.minT = min_t
        self.n_splits = n_splits
        self.folds = None
        self.c_cur_best = float("-inf")

    def estimate(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        self.folds = np.random.randint(5, size=len(data))
        gamma = labels.sum() / len(labels)
        labels = bitarray((labels == 1).tolist())
        c_its_ests = []
        c_estimate = 0.5
        for it in range(self.nbIterations):
            c_estimates = []

            for tree_train, estimate in self.generate_folds(self.folds):
                self.c_cur_best = self.low_c(
                    estimate, labels, 1.0, self.minT, c=c_estimate
                )
                cur_delta = (
                    self.delta if self.delta else self.pick_delta(estimate.count())
                )

                if self.useMostPromisingOnly:
                    c_tree_best = 0.0
                    most_promising = estimate
                    for tree_subset, estimate_subset in self.subsetsThroughDT(
                        data,
                        tree_train,
                        estimate,
                        labels,
                        splitCrit=self.max_bepp(self.k),
                        minExamples=self.minT,
                        maxSplits=self.maxSplits,
                        c_prior=c_estimate,
                        delta=cur_delta,
                        n_splits=self.n_splits,
                    ):
                        tree_est_here = self.low_c(
                            tree_subset, labels, cur_delta, 1, c=c_estimate
                        )
                        if tree_est_here > c_tree_best:
                            c_tree_best = tree_est_here
                            most_promising = estimate_subset

                    c_estimates.append(
                        max(
                            self.c_cur_best,
                            self.low_c(
                                most_promising,
                                labels,
                                cur_delta,
                                self.minT,
                                c=c_estimate,
                            ),
                        )
                    )
                else:
                    for tree_subset, estimate_subset in self.subsetsThroughDT(
                        data,
                        tree_train,
                        estimate,
                        labels,
                        splitCrit=self.max_bepp(self.k),
                        minExamples=self.minT,
                        maxSplits=self.maxSplits,
                        c_prior=c_estimate,
                        delta=cur_delta,
                        n_splits=self.n_splits,
                    ):
                        est_here = self.low_c(
                            estimate_subset, labels, cur_delta, self.minT, c=c_estimate
                        )
                        self.c_cur_best = max(self.c_cur_best, est_here)
                    c_estimates.append(self.c_cur_best)

            c_estimate = sum(c_estimates) / float(len(c_estimates))
            c_its_ests.append(c_estimates)

        alpha = self.tice_c_to_alpha(c_estimate, gamma)
        return {"alpha": alpha}

    def tice_c_to_alpha(self, c, gamma):
        return max(0.0, 1 - (1 - gamma) * (1 - c) / gamma / c)

    def subsetsThroughDT(
        self,
        data,
        tree_train,
        estimate,
        labels,
        splitCrit,
        minExamples=10,
        maxSplits=500,
        c_prior=0.5,
        delta=0.0,
        n_splits=3,
    ):
        """
        This learns a decision tree and updates the label frequency lower bound for every tried split.
        It splits every variable into 4 pieces: [0,.25[ , [.25, .5[ , [.5,.75[ , [.75,1]
        The input data is expected to have only binary or continues variables with values between 0 and 1.
        To achieve this, the multivalued variables should be binarized and the continuous variables should be normalized
        Max: Return all the subsets encountered
        """

        all_data = tree_train | estimate
        borders = np.linspace(0, 1, n_splits + 2, True).tolist()[1:-1]

        def makeSubsets(a):
            subsets = []
            options = bitarray(all_data)
            for b in borders:
                X_cond = bitarray((data[:, a] < b).tolist()) & options
                options &= ~X_cond
                subsets.append(X_cond)
            subsets.append(options)
            return subsets

        conditionSets = [makeSubsets(a) for a in range(data.shape[1])]

        priorityq = []
        heapq.heappush(
            priorityq,
            (
                -self.low_c(tree_train, labels, delta, 0, c=c_prior),
                -(tree_train & labels).count(),
                tree_train,
                estimate,
                set(range(data.shape[1])),
                0,
            ),
        )
        yield (tree_train, estimate)

        n = 0
        minimumLabeled = 1
        while n < maxSplits and len(priorityq) > 0:
            n += 1
            (ppos, neg_lab_count, subset_train, subset_estimate, available, depth) = (
                heapq.heappop(priorityq)
            )
            lab_count = -neg_lab_count

            best_a = -1
            best_score = -1
            best_subsets_train = []
            best_subsets_estimate = []
            best_lab_counts = []
            uselessAs = set()

            for a in available:
                subsets_train = list(
                    map(lambda X_cond: X_cond & subset_train, conditionSets[a])
                )
                subsets_estimate = list(
                    map(lambda X_cond: X_cond & subset_estimate, conditionSets[a])
                )  # X_cond & subset_train
                estimate_lab_counts = list(
                    map(lambda subset: (subset & labels).count(), subsets_estimate)
                )
                if max(estimate_lab_counts) < minimumLabeled:
                    uselessAs.add(a)
                else:
                    score = splitCrit(
                        list(
                            map(
                                lambda subsub: (
                                    subsub.count(),
                                    (subsub & labels).count(),
                                ),
                                subsets_train,
                            )
                        )
                    )
                    if score > best_score:
                        best_score = score
                        best_a = a
                        best_subsets_train = subsets_train
                        best_subsets_estimate = subsets_estimate
                        best_lab_counts = estimate_lab_counts

            fake_split = (
                len(
                    list(
                        filter(lambda subset: subset.count() > 0, best_subsets_estimate)
                    )
                )
                == 1
            )

            if best_score > 0 and not fake_split:
                newAvailable = available - {best_a} - uselessAs
                for subsub_train, subsub_estimate in zip(
                    best_subsets_train, best_subsets_estimate
                ):
                    yield (subsub_train, subsub_estimate)
                minimumLabeled = (
                    c_prior
                    * (1 - c_prior)
                    * (1 - delta)
                    / (delta * (1 - self.c_cur_best) ** 2)
                )

                for subsub_lab_count, subsub_train, subsub_estimate in zip(
                    best_lab_counts, best_subsets_train, best_subsets_estimate
                ):
                    if subsub_lab_count > minimumLabeled:
                        total = subsub_train.count()
                        if (
                            total > minExamples
                        ):  # stop criterion: minimum size for splitting
                            train_lab_count = (subsub_train & labels).count()
                            if (
                                lab_count != 0 and lab_count != total
                            ):  # stop criterion: purity
                                heapq.heappush(
                                    priorityq,
                                    (
                                        -self.low_c(
                                            subsub_train, labels, delta, 0, c=c_prior
                                        ),
                                        -train_lab_count,
                                        subsub_train,
                                        subsub_estimate,
                                        newAvailable,
                                        depth + 1,
                                    ),
                                )

    def pick_delta(self, T):
        return max(0.025, 1 / (1 + 0.004 * T))

    def low_c(self, data, label, delta, minT, c=0.5):
        T = float(data.count())
        if T < minT:
            return 0.0
        L = float((data & label).count())
        clow = L / T - math.sqrt(c * (1 - c) * (1 - delta) / (delta * T))
        return clow

    def max_bepp(self, k):
        def fun(counts):
            return max(
                list(
                    map(
                        lambda T_P: (
                            0 if T_P[0] == 0 else float(T_P[1]) / (T_P[0] + k)
                        ),
                        counts,
                    )
                )
            )

        return fun

    def generate_folds(self, folds):
        for fold in range(max(folds) + 1):
            tree_train = bitarray((folds == fold).tolist())
            estimate = ~tree_train
            yield (tree_train, estimate)
