import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, KFold
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from random import sample
from typing import Dict


class MonotonizingTrends:
    def __init__(self, a=None, MT_coef=1):
        self.counter = dict()
        self.array_new = []
        if a is None:
            self.array_old = []
        else:
            self.add_array(a)
        self.MT_coef = MT_coef

    def add_array(self, a):
        if isinstance(a, np.ndarray) or isinstance(a, pd.Series):
            a = a.tolist()
        self.array_old = a

    def reset(self):
        self.counter = dict()
        self.array_old = []
        self.array_new = []

    def get_highest_point(self):
        if self.counter:
            return max(self.counter)
        else:
            return np.nan

    def add_point_to_counter(self, point):
        if point not in self.counter.keys():
            self.counter[point] = 1

    def change_counter_according_to_point(self, point):
        for key in self.counter.keys():
            if key <= point:
                self.counter[key] += 1
            else:
                self.counter[key] -= self.MT_coef

    def clear_counter(self):
        for key, value in list(self.counter.items()):
            if value <= 0:
                self.counter.pop(key)

    def update_counter_with_point(self, point):
        self.change_counter_according_to_point(point)
        self.clear_counter()
        self.add_point_to_counter(point)

    def monotonize_point(self, point=None):
        if point is None:
            point = self.array_old.pop(0)
        new_point = max(point, self.get_highest_point())
        self.array_new.append(new_point)
        self.update_counter_with_point(point)
        return new_point

    def monotonize_array(self, a=None, reset=False, decay_MT_coef=False):
        if a is not None:
            self.add_array(a)
        decay_by = 0
        if decay_MT_coef:
            decay_by = self.MT_coef / len(a)

        for _ in range(len(self.array_old)):
            self.monotonize_point()
            if decay_MT_coef:
                self.MT_coef -= decay_by

        if not reset:
            return self.array_new
        else:
            array_new = self.array_new[:]
            self.reset()
            return array_new


class DEDPUL:
    def __init__(self, bayes: bool = False):
        self.bayes = bayes

    def estimate(self, X, y) -> Dict[str, float]:
        preds = self.estimate_preds_cv(
            X=X,
            target=y,
            alpha=None,
            training_mode="standard",
            bayes=self.bayes,
            train_nn_options=dict(),
        )
        if self.bayes:
            preds, means, variances = preds
            diff = self.estimate_diff_bayes(means, variances, y)
        else:
            diff = self.estimate_diff(preds, y)

        alpha, _ = self.estimate_poster_em(diff=diff, mode="dedpul", alpha=None)
        return {"alpha": float(alpha)}

    def estimate_poster_em(
        self,
        diff=None,
        preds=None,
        target=None,
        mode="dedpul",
        converge=True,
        tol=10**-5,
        max_iterations=1000,
        nonconverge=True,
        step=0.001,
        max_diff=0.05,
        plot=False,
        disp=False,
        alpha=None,
        alpha_as_mean_poster=True,
        **kwargs
    ):
        """
        Performs Expectation-Maximization to estimate posteriors and priors alpha (if not provided) of N in U
            with either of 'en' or 'dedpul' methods; both 'converge' and 'nonconverge' are recommended to be set True for
            better estimate
        :param diff: difference of densities f_p/f_u for the sample U, np.array (n,), output of estimate_diff()
        :param preds: predictions of classifier, np.array with shape (n,)
        :param target: binary vector, 0 if positive, 1 if unlabeled, np.array with shape (n,)
        :param mode: 'dedpul' or 'en'; if 'dedpul', diff needs to be provided; if 'en', preds and target need to be provided
        :param converge: True or False; True if convergence estimate should be computed
        :param tol: tolerance of error between priors and mean posteriors, indicator of convergence
        :param max_iterations: if exceeded, search of converged alpha stops even if tol is not reached
        :param nonconverge: True or False; True if non-convergence estimate should be computed
        :param step: gap between points of the [0, 1, step] gird to choose best alpha from
        :param max_diff: alpha with difference of mean posteriors and priors bigger than max_diff cannot be chosen;
            an heuristic to choose bigger alpha
        :param plot: True or False, if True - plots ([0, 1, grid], mean posteriors - alpha) and
            ([0, 1, grid], second lag of (mean posteriors - alpha))
        :param disp: True or False, if True - displays if the algorithm didn't converge
        :param alpha: proportions of N in U; is estimated if None
        :return: tuple (alpha, poster), e.g. (priors, posteriors) of N in U for the U sample
        """
        if alpha is not None:
            alpha, poster = self.estimate_poster_dedpul(
                diff,
                alpha=alpha,
                alpha_as_mean_poster=alpha_as_mean_poster,
                tol=tol,
                **kwargs
            )
            return alpha, poster

        alpha_converge = 0
        for i in range(max_iterations):
            _, poster_converge = self.estimate_poster_dedpul(
                diff, alpha=alpha_converge, **kwargs
            )
            mean_poster = np.mean(poster_converge)
            error = mean_poster - alpha_converge

            if np.abs(error) < tol:
                break
            if np.min(poster_converge) > 0:
                break
            alpha_converge = mean_poster

        errors = np.array([])
        for alpha_nonconverge in np.arange(0, 1, step):
            _, poster_nonconverge = self.estimate_poster_dedpul(
                diff, alpha=alpha_nonconverge, **kwargs
            )
            errors = np.append(errors, np.mean(poster_nonconverge) - alpha_nonconverge)

        idx = np.argmax(np.diff(np.diff(errors))[errors[1:-1] < max_diff])
        alpha_nonconverge = np.arange(0, 1, step)[1:-1][errors[1:-1] < max_diff][idx]
        if (alpha_nonconverge >= alpha_converge) or (  # converge and nonconverge and
            ((errors < 0).sum() > 1) and (alpha_converge < 1 - step)
        ):
            return alpha_converge, poster_converge

        elif nonconverge:
            _, poster_nonconverge = self.estimate_poster_dedpul(
                diff, alpha=alpha_nonconverge, **kwargs
            )
            return alpha_nonconverge, poster_nonconverge
        else:
            return None, None

    def estimate_poster_dedpul(
        self,
        diff,
        alpha=None,
        quantile=0.05,
        alpha_as_mean_poster=False,
        max_it=100,
        **kwargs
    ):
        """
        Estimates posteriors and priors alpha (if not provided) of N in U with dedpul method
        :param diff: difference of densities f_p / f_u for the sample U, np.array (n,), output of estimate_diff()
        :param alpha: priors, share of N in U (estimated if None)
        :param quantile: if alpha is None, relaxation of the estimate of alpha;
            here alpha is estimaeted as infinum, and low quantile is its relaxed version;
            share of posteriors probabilities that we allow to be negative (with the following zeroing-out)
        :param kwargs: dummy

        :return: tuple (alpha, poster), e.g. (priors, posteriors) of N in U for the U sample, represented by diff
        """
        if alpha_as_mean_poster and (alpha is not None):
            poster = 1 - diff * (1 - alpha)
            poster[poster < 0] = 0
            cur_alpha = np.mean(poster)
            if cur_alpha < alpha:
                left_border = alpha
                right_border = 1
            else:
                left_border = 0
                right_border = alpha

                poster_zero = 1 - diff
                poster_zero[poster_zero < 0] = 0
                if np.mean(poster_zero) > alpha:
                    left_border = -50
                    right_border = 0
            it = 0
            try_alpha = cur_alpha
            while (abs(cur_alpha - alpha) > kwargs.get("tol", 10**-5)) and (
                it < max_it
            ):
                try_alpha = left_border + (right_border - left_border) / 2
                poster = 1 - diff * (1 - try_alpha)
                poster[poster < 0] = 0
                cur_alpha = np.mean(poster)
                if cur_alpha > alpha:
                    right_border = try_alpha
                else:
                    left_border = try_alpha
                it += 1
            alpha = try_alpha
            if it >= max_it:
                print(
                    "Exceeded maximal number of iterations in finding mean_poster=alpha"
                )
        else:
            if alpha is None:
                alpha = 1 - 1 / max(
                    np.quantile(diff, 1 - quantile, interpolation="higher"), 1
                )
            poster = 1 - diff * (1 - alpha)
            poster[poster < 0] = 0
        return alpha, poster

    def estimate_preds_cv(
        self,
        X,
        target,
        cv=3,
        n_networks=1,
        lr=1e-4,
        hid_dim=32,
        n_hid_layers=1,
        random_state=None,
        training_mode="standard",
        alpha=None,
        l2=1e-4,
        train_nn_options=None,
        bayes=False,
        bn=True,
    ):
        if train_nn_options is None:
            train_nn_options = dict()

        preds = np.zeros(
            (
                n_networks,
                X.shape[0],
            )
        )
        means = np.zeros(
            (
                n_networks,
                X.shape[0],
            )
        )
        variances = np.zeros(
            (
                n_networks,
                X.shape[0],
            )
        )

        for i in range(n_networks):
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            for train_index, test_index in kf.split(X, target):
                train_data = X[train_index]
                train_target = target[train_index]
                mix_data = train_data[train_target == 1]
                pos_data = train_data[train_target == 0]
                test_data = X[test_index]
                test_target = target[test_index]

                mix_data_test = test_data[test_target == 1]
                pos_data_test = test_data[test_target == 0]
                discriminator = Net(
                    inp_dim=X.shape[1],
                    out_dim=1,
                    hid_dim=hid_dim,
                    n_hid_layers=n_hid_layers,
                    bayes=bayes,
                    bn=bn,
                )
                d_optimizer = optim.Adam(
                    discriminator.parameters(), lr=lr, weight_decay=l2
                )

                train_NN(
                    mix_data,
                    pos_data,
                    discriminator,
                    d_optimizer,
                    mix_data_test,
                    pos_data_test,
                    nnre_alpha=alpha,
                    d_scheduler=None,
                    training_mode=training_mode,
                    bayes=bayes,
                    **train_nn_options
                )
                if bayes:
                    pred, mean, var = discriminator(
                        torch.as_tensor(test_data, dtype=torch.float32),
                        return_params=True,
                        sample_noise=False,
                    )
                    (
                        preds[i, test_index],
                        means[i, test_index],
                        variances[i, test_index],
                    ) = (
                        pred.detach().numpy().flatten(),
                        mean.detach().numpy().flatten(),
                        var.detach().numpy().flatten(),
                    )
                else:
                    preds[i, test_index] = (
                        discriminator(torch.as_tensor(test_data, dtype=torch.float32))
                        .detach()
                        .numpy()
                        .flatten()
                    )

        preds = preds.mean(axis=0)
        if bayes:
            means, variances = means.mean(axis=0), variances.mean(axis=0)
            return preds, means, variances
        else:
            return preds

    def estimate_diff(
        self,
        preds,
        target,
        bw_mix=0.05,
        bw_pos=0.1,
        kde_mode="logit",
        threshold=None,
        k_neighbours=None,
        tune=False,
        MT=True,
        MT_coef=0.2,
        decay_MT_coef=False,
        kde_type="kde",
        n_gauss_mix=20,
        n_gauss_pos=10,
        bins_mix=20,
        bins_pos=20,
    ):
        """
        Estimates densities of predictions y(x) for P and U and ratio between them f_p / f_u for U sample;
            uses kernel density estimation (kde);
            post-processes difference of estimated densities - imposes monotonicity on lower preds
            (so that diff is partly non-decreasing) and applies rolling median to further reduce variance
        :param preds: predictions of NTC y(x), probability of belonging to U rather than P, np.array with shape (n,)
        :param target: binary vector, 0 if positive, 1 if unlabeled, np.array with shape (n,)
        :param bw_mix: bandwidth for kde of U
        :param bw_pos: bandwidth for kde of P
        :param kde_mode: 'prob', 'log_prob' or 'logit'; default is 'logit'
        :param monotonicity: monotonicity is imposed on density difference for predictions below this number, float in [0, 1]
        :param k_neighbours: difference is relaxed with median rolling window with size k_neighbours * 2 + 1,
            default = int(preds[target == 1].shape[0] // 10)

        :return: difference of densities f_p / f_u for U sample
        """

        if kde_mode is None:
            kde_mode = "logit"

        if (threshold is None) or (threshold == "mid"):
            threshold = preds[target == 1].mean() / 2 + preds[target == 0].mean() / 2
        elif threshold == "low":
            threshold = preds[target == 0].mean()
        elif threshold == "high":
            threshold = preds[target == 1].mean()

        if k_neighbours is None:
            k_neighbours = int(preds[target == 1].shape[0] // 20)

        if kde_mode == "prob":
            kde_inner_fun = lambda x: x
            kde_outer_fun = lambda dens, x: dens(x)
        elif kde_mode == "log_prob":
            kde_inner_fun = lambda x: np.log(x)
            kde_outer_fun = lambda dens, x: dens(np.log(x)) / (x + 10**-5)
        elif kde_mode == "logit":
            kde_inner_fun = lambda x: np.log(x / (1 - x + 10**-5))
            kde_outer_fun = lambda dens, x: dens(np.log(x / (1 - x + 10**-5))) / (
                x * (1 - x) + 10**-5
            )

        if kde_type == "kde":
            if tune:
                bw_mix = maximize_log_likelihood(
                    preds[target == 1], kde_inner_fun, kde_outer_fun, kde_type=kde_type
                )
                bw_pos = maximize_log_likelihood(
                    preds[target == 0], kde_inner_fun, kde_outer_fun, kde_type=kde_type
                )

            kde_mix = gaussian_kde(
                np.apply_along_axis(kde_inner_fun, 0, preds[target == 1]), bw_mix
            )
            kde_pos = gaussian_kde(
                np.apply_along_axis(kde_inner_fun, 0, preds[target == 0]), bw_pos
            )

        elif kde_type == "GMM":
            if tune:
                n_gauss_mix = maximize_log_likelihood(
                    preds[target == 1], kde_inner_fun, kde_outer_fun, kde_type=kde_type
                )
                n_gauss_pos = maximize_log_likelihood(
                    preds[target == 0], kde_inner_fun, kde_outer_fun, kde_type=kde_type
                )

            GMM_mix = GaussianMixture(n_gauss_mix, covariance_type="spherical").fit(
                np.apply_along_axis(kde_inner_fun, 0, preds[target == 1]).reshape(-1, 1)
            )
            GMM_pos = GaussianMixture(n_gauss_pos, covariance_type="spherical").fit(
                np.apply_along_axis(kde_inner_fun, 0, preds[target == 0]).reshape(-1, 1)
            )

            kde_mix = lambda x: np.exp(GMM_mix.score_samples(x.reshape(-1, 1)))
            kde_pos = lambda x: np.exp(GMM_pos.score_samples(x.reshape(-1, 1)))

        elif kde_type == "hist":
            if tune:
                bins_mix = maximize_log_likelihood(
                    preds[target == 1],
                    kde_inner_fun,
                    lambda kde, x: kde(x),
                    kde_type=kde_type,
                )
                bins_pos = maximize_log_likelihood(
                    preds[target == 0],
                    kde_inner_fun,
                    lambda kde, x: kde(x),
                    kde_type=kde_type,
                )
            bars_mix = np.histogram(
                preds[target == 1], bins=bins_mix, range=(0, 1), density=True
            )[0]
            bars_pos = np.histogram(
                preds[target == 0], bins=bins_pos, range=(0, 1), density=True
            )[0]

            kde_mix = lambda x: bars_mix[
                np.clip((x // (1 / bins_mix)).astype(int), 0, bins_mix - 1)
            ]
            kde_pos = lambda x: bars_pos[
                np.clip((x // (1 / bins_pos)).astype(int), 0, bins_pos - 1)
            ]
            kde_outer_fun = lambda kde, x: kde(x)

        # sorting to relax and impose monotonicity
        sorted_mixed = np.sort(preds[target == 1])

        diff = np.apply_along_axis(
            lambda x: kde_outer_fun(kde_pos, x) / (kde_outer_fun(kde_mix, x) + 10**-5),
            axis=0,
            arr=sorted_mixed,
        )
        diff[diff > 50] = 50
        diff = rolling_apply(diff, 5)
        diff = np.append(
            np.flip(
                np.maximum.accumulate(np.flip(diff[sorted_mixed <= threshold], axis=0)),
                axis=0,
            ),
            diff[sorted_mixed > threshold],
        )
        diff = rolling_apply(diff, k_neighbours)

        if MT:
            MTrends = MonotonizingTrends(MT_coef=MT_coef)
            diff = np.flip(
                np.array(
                    MTrends.monotonize_array(
                        np.flip(diff, axis=0), reset=True, decay_MT_coef=decay_MT_coef
                    )
                ),
                axis=0,
            )

        diff.sort()
        diff = np.flip(diff, axis=0)
        # desorting
        diff = diff[np.argsort(np.argsort(preds[target == 1]))]
        return diff

    def estimate_diff_bayes(
        self, means, variances, target, threshold=None, k_neighbours=None
    ):
        if threshold == "mid":
            threshold = means[target == 1].mean() / 2 + means[target == 0].mean() / 2
        elif (threshold == "low") or (threshold is None):
            threshold = means[target == 0].mean()
        elif threshold == "high":
            threshold = means[target == 1].mean()

        if k_neighbours is None:
            k_neighbours = int(means[target == 1].shape[0] // 20)

        n_mix = means[target == 1].shape[0]
        GMM_mix = GaussianMixtureNoFit(
            n_mix,
            covariance_type="spherical",
            max_iter=1,
            n_init=1,
            weights_init=np.ones(n_mix) / n_mix,
            means_init=means[target == 1].reshape(-1, 1),
            precisions_init=1 / variances[target == 1],
        ).fit(means[target == 1].reshape(-1, 1))
        kde_mix = lambda x: np.exp(GMM_mix.score_samples(x))

        n_pos = means[target == 0].shape[0]
        GMM_pos = GaussianMixtureNoFit(
            n_pos,
            covariance_type="spherical",
            max_iter=1,
            n_init=1,
            weights_init=np.ones(n_pos) / n_pos,
            means_init=means[target == 0].reshape(-1, 1),
            precisions_init=1 / variances[target == 0],
        ).fit(means[target == 0].reshape(-1, 1))
        kde_pos = lambda x: np.exp(GMM_pos.score_samples(x))

        sorted_means = np.sort(means[target == 1])
        # diff = np.array(kde_pos(sorted_means.reshape(-1, 1)) / kde_mix(sorted_means.reshape(-1, 1)))
        diff = np.array([])
        for i in range(int(np.ceil(len(sorted_means) / 1000))):
            current = sorted_means[i * 1000 : min((i + 1) * 1000, len(sorted_means))]
            diff = np.append(
                diff, kde_pos(current.reshape(-1, 1)) / kde_mix(current.reshape(-1, 1))
            )
        diff[diff > 50] = 50

        diff = rolling_apply(diff, k_neighbours)
        diff = np.append(
            np.flip(
                np.maximum.accumulate(np.flip(diff[sorted_means <= threshold], axis=0)),
                axis=0,
            ),
            diff[sorted_means > threshold],
        )

        diff = diff[np.argsort(np.argsort(means[target == 1]))]
        return diff


class Net(nn.Module):
    def __init__(
        self, inp_dim, out_dim=1, hid_dim=32, n_hid_layers=1, bayes=False, bn=True
    ):
        super(Net, self).__init__()
        self.bayes = bayes
        self.bn = bn
        self.n_hid_layers = n_hid_layers

        self.inp = nn.Linear(inp_dim, hid_dim, bias=not bn)
        if bn:
            self.inp_bn = nn.BatchNorm1d(hid_dim, momentum=0.1)
        if self.n_hid_layers > 0:
            self.hid = nn.Sequential()
            for i in range(n_hid_layers):
                self.hid.add_module(str(i), nn.Linear(hid_dim, hid_dim, bias=not bn))
                if bn:
                    self.hid.add_module(
                        "bn" + str(i), nn.BatchNorm1d(hid_dim, momentum=0.1)
                    )
                self.hid.add_module("a" + str(i), nn.ReLU())

        if self.bayes:
            self.out_mean = nn.Linear(hid_dim, out_dim)
            self.out_logvar = nn.Linear(hid_dim, out_dim)
        else:
            self.out = nn.Linear(hid_dim, out_dim)

    def forward(self, x, return_params=False, sample_noise=False):
        if self.bn:
            x = F.relu(self.inp_bn(self.inp(x)))
        else:
            x = F.relu(self.inp(x))
        if self.n_hid_layers > 0:
            x = self.hid(x)

        if self.bayes:
            mean, logvar = self.out_mean(x), self.out_logvar(x)
            var = torch.exp(logvar * 0.5)
            if sample_noise:
                x = mean + var * torch.randn_like(var)
            else:
                x = mean
        else:
            mean = self.out(x)
            var = torch.zeros_like(mean) + 1e-3
            x = mean
        p = F.sigmoid(x)

        if return_params:
            return p, mean, var
        else:
            return p


def d_loss_standard(batch_mix, batch_pos, discriminator, loss_function=None):
    n_mix = batch_mix.shape[0]
    preds = discriminator(torch.cat([batch_mix, batch_pos]))
    d_mix, d_pos = preds[:n_mix], preds[n_mix:]
    if (loss_function is None) or (loss_function == "log"):
        loss_function = lambda x: torch.log(x + 10**-5)  # log loss
    elif loss_function == "sigmoid":
        loss_function = lambda x: x  # sigmoid loss
    elif loss_function == "brier":
        loss_function = lambda x: x**2  # brier loss
    return (
        -(torch.mean(loss_function(1 - d_pos)) + torch.mean(loss_function(d_mix))) / 2
    )


def KL_normal(m, v):
    return (torch.log(1 / (v.sqrt() + 1e-6)) + (m**2 + v - 1) * 0.5).mean()


def d_loss_bayes(batch_mix, batch_pos, discriminator, loss_function=None, w=1e-4):
    n_mix = batch_mix.shape[0]
    preds, means, var = discriminator(
        torch.cat([batch_mix, batch_pos]), return_params=True, sample_noise=True
    )
    d_mix, d_pos, mean_mix, mean_pos, var_mix, var_pos = (
        preds[:n_mix],
        preds[n_mix:],
        means[:n_mix],
        means[n_mix:],
        var[:n_mix],
        var[n_mix:],
    )
    if (loss_function is None) or (loss_function == "log"):
        loss_function = lambda x: torch.log(x + 10**-5)  # log loss
    elif loss_function == "sigmoid":
        loss_function = lambda x: x  # sigmoid loss
    elif loss_function == "brier":
        loss_function = lambda x: x**2  # brier loss
    loss = (
        -(torch.mean(loss_function(1 - d_pos)) + torch.mean(loss_function(d_mix))) / 2
    )
    loss += (KL_normal(mean_mix, var_mix) + KL_normal(mean_pos, var_pos)) / 2 * w
    return loss


def d_loss_nnRE(
    batch_mix, batch_pos, discriminator, alpha, beta=0.0, gamma=1.0, loss_function=None
):
    n_mix = batch_mix.shape[0]
    preds = discriminator(torch.cat([batch_mix, batch_pos]))
    d_mix, d_pos = preds[:n_mix], preds[n_mix:]
    if (loss_function is None) or (loss_function == "brier"):
        loss_function = lambda x: (1 - x) ** 2  # brier loss
    elif loss_function == "sigmoid":
        loss_function = lambda x: 1 - x  # sigmoid loss
    elif loss_function in {"log", "logistic"}:
        loss_function = lambda x: torch.log(1 - x + 10**-5)  # log loss
    pos_part = (1 - alpha) * torch.mean(loss_function(1 - d_pos))
    nn_part = torch.mean(loss_function(d_mix)) - (1 - alpha) * torch.mean(
        loss_function(d_pos)
    )

    if nn_part.item() >= -beta:
        return pos_part + nn_part
    else:
        return -nn_part * gamma


def compute_log_likelihood(preds, kde, kde_outer_fun=lambda kde, x: kde(x)):
    likelihood = np.apply_along_axis(lambda x: kde_outer_fun(kde, x), 0, preds)
    return np.log(likelihood).mean()


def maximize_log_likelihood(
    preds,
    kde_inner_fun,
    kde_outer_fun,
    n_folds=5,
    kde_type="kde",
    bw_low=0.01,
    bw_high=0.4,
    n_gauss_low=1,
    n_gauss_high=50,
    bins_low=20,
    bins_high=250,
    n_steps=25,
):
    kf = KFold(n_folds, shuffle=True)
    idx_best, like_best = 0, 0
    bws = np.exp(np.linspace(np.log(bw_low), np.log(bw_high), n_steps))
    n_gauss = np.linspace(n_gauss_low, n_gauss_high, n_steps).astype(int)
    bins = np.linspace(bins_low, bins_high, n_steps).astype(int)
    for idx, (bw, n_g, bin) in enumerate(zip(bws, n_gauss, bins)):
        like = 0
        for train_idx, test_idx in kf.split(preds):
            if kde_type == "kde":
                kde = gaussian_kde(
                    np.apply_along_axis(kde_inner_fun, 0, preds[train_idx]), bw
                )
            elif kde_type == "GMM":
                GMM = GaussianMixture(n_g, covariance_type="spherical").fit(
                    np.apply_along_axis(kde_inner_fun, 0, preds[train_idx]).reshape(
                        -1, 1
                    )
                )
                kde = lambda x: np.exp(GMM.score_samples(x.reshape(-1, 1)))
            elif kde_type == "hist":
                bars = np.histogram(
                    preds[train_idx], bins=bin, range=(0, 1), density=True
                )[0]
                kde = lambda x: bars[np.clip((x // (1 / bin)).astype(int), 0, bin - 1)]
                kde_outer_fun = lambda kde, x: kde(x)

            like += compute_log_likelihood(preds[test_idx], kde, kde_outer_fun)
        if like > like_best:
            like_best, idx_best = like, idx
    if kde_type == "kde":
        return bws[idx_best]
    elif kde_type == "GMM":
        return n_gauss[idx_best]
    elif kde_type == "hist":
        return bins[idx_best]


def train_NN(
    mix_data,
    pos_data,
    discriminator,
    d_optimizer,
    mix_data_test=None,
    pos_data_test=None,
    n_epochs=200,
    batch_size=64,
    n_batches=None,
    n_early_stop=5,
    d_scheduler=None,
    training_mode="standard",
    disp=False,
    loss_function=None,
    nnre_alpha=None,
    metric=None,
    stop_by_metric=False,
    bayes=False,
    bayes_weight=1e-5,
    beta=0,
    gamma=1,
):
    """
    Train discriminator to classify mix_data from pos_data.
    """
    d_losses_train = []
    d_losses_test = []
    d_metrics_test = []
    if n_batches is None:
        n_batches = min(
            int(mix_data.shape[0] / batch_size), int(pos_data.shape[0] / batch_size)
        )
        batch_size_mix = batch_size_pos = batch_size
    else:
        batch_size_mix, batch_size_pos = int(mix_data.shape[0] / n_batches), int(
            pos_data.shape[0] / n_batches
        )
    if mix_data_test is not None:
        data_test = np.concatenate((pos_data_test, mix_data_test))
        target_test = np.concatenate(
            (np.zeros((pos_data_test.shape[0],)), np.ones((mix_data_test.shape[0],)))
        )

    for epoch in range(n_epochs):

        discriminator.train()

        d_losses_cur = []
        if d_scheduler is not None:
            d_scheduler.step()

        for i in range(n_batches):

            batch_mix = np.array(sample(list(mix_data), batch_size_mix))
            batch_pos = np.array(sample(list(pos_data), batch_size_pos))

            batch_mix = torch.as_tensor(batch_mix, dtype=torch.float32)
            batch_pos = torch.as_tensor(batch_pos, dtype=torch.float32)

            # Optimize D
            d_optimizer.zero_grad()

            if training_mode == "standard":
                if bayes:
                    loss = d_loss_bayes(
                        batch_mix, batch_pos, discriminator, loss_function, bayes_weight
                    )
                else:
                    loss = d_loss_standard(
                        batch_mix, batch_pos, discriminator, loss_function
                    )

            else:
                loss = d_loss_nnRE(
                    batch_mix,
                    batch_pos,
                    discriminator,
                    nnre_alpha,
                    beta=beta,
                    gamma=gamma,
                    loss_function=loss_function,
                )

            loss.backward()
            d_optimizer.step()
            d_losses_cur.append(loss.cpu().item())

        d_losses_train.append(round(np.mean(d_losses_cur).item(), 5))

        if mix_data_test is not None and pos_data_test is not None:

            discriminator.eval()

            if training_mode == "standard":
                if bayes:
                    d_losses_test.append(
                        round(
                            d_loss_bayes(
                                torch.as_tensor(mix_data_test, dtype=torch.float32),
                                torch.as_tensor(pos_data_test, dtype=torch.float32),
                                discriminator,
                                w=bayes_weight,
                            ).item(),
                            5,
                        )
                    )
                else:
                    d_losses_test.append(
                        round(
                            d_loss_standard(
                                torch.as_tensor(mix_data_test, dtype=torch.float32),
                                torch.as_tensor(pos_data_test, dtype=torch.float32),
                                discriminator,
                            ).item(),
                            5,
                        )
                    )
            elif training_mode == "nnre":
                d_losses_test.append(
                    round(
                        d_loss_nnRE(
                            torch.as_tensor(mix_data_test, dtype=torch.float32),
                            torch.as_tensor(pos_data_test, dtype=torch.float32),
                            discriminator,
                            nnre_alpha,
                        ).item(),
                        5,
                    )
                )
            if metric is not None:
                d_metrics_test.append(
                    metric(
                        target_test,
                        discriminator(torch.as_tensor(data_test, dtype=torch.float32))
                        .detach()
                        .numpy(),
                    )
                )

            if disp:
                if not metric:
                    print(
                        "epoch",
                        epoch,
                        ", train_loss=",
                        d_losses_train[-1],
                        ", test_loss=",
                        d_losses_test[-1],
                    )
                else:
                    print(
                        "epoch",
                        epoch,
                        ", train_loss=",
                        d_losses_train[-1],
                        ", test_loss=",
                        d_losses_test[-1],
                        "test_metric=",
                        d_metrics_test[-1],
                    )

            if epoch >= n_early_stop:
                if_stop = True
                for i in range(n_early_stop):
                    if metric is not None and stop_by_metric:
                        if d_metrics_test[-i - 1] < d_metrics_test[-n_early_stop - 1]:
                            if_stop = False
                            break
                    else:
                        if d_losses_test[-i - 1] < d_losses_test[-n_early_stop - 1]:
                            if_stop = False
                            break
                if if_stop:
                    break
        elif disp:
            pass
            # print("epoch", epoch, ", train_loss=", d_losses_train[-1])

    discriminator.eval()

    return d_losses_train, d_losses_test


class GaussianMixtureNoFit(GaussianMixture):
    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
        max_components=None,
    ):

        self.max_components = max_components

        idx = np.arange(means_init.shape[0])
        if (max_components is not None) and (means_init.shape[0] > max_components):
            n_components = min(n_components, max_components)
            np.random.shuffle(idx)
            idx = idx[: self.max_components]

        weights_init = weights_init[idx]
        weights_init /= weights_init.sum()
        means_init = means_init[idx]
        precisions_init = precisions_init[idx]

        super().__init__(
            n_components,
            covariance_type,
            tol,
            reg_covar,
            max_iter,
            n_init,
            init_params,
            weights_init,
            means_init,
            precisions_init,
            random_state,
            warm_start,
            verbose,
            verbose_interval,
        )

        self.weights = weights_init
        self.means = means_init
        self.precisions = precisions_init
        self.weights_ = weights_init
        self.means_ = means_init
        self.precisions_ = precisions_init
        self.precisions_cholesky_ = np.sqrt(precisions_init)
        self.covariances_ = 1 / precisions_init
        self.covariances = 1 / precisions_init

    def _initialize(self, X, resp):
        pass

    def _m_step(self, X, log_resp):
        pass

    def fit(self, X, y=None):
        pass
        return self


def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))


def rolling_apply(diff, k_neighbours):
    s = pd.Series(diff)
    s = np.concatenate(
        (
            s.iloc[: 2 * k_neighbours].expanding().median()[::2].values,
            s.rolling(k_neighbours * 2 + 1, center=True).median().dropna().values,
            np.flip(
                np.flip(s.iloc[-2 * k_neighbours :], axis=0).expanding().median()[::2],
                axis=0,
            ).values,
        )
    )
    return s


if __name__ == "__main__":
    data = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    alg = DEDPUL()
    print(alg.estimate(data, y))
