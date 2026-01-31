import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class AlphaMax:
    def __init__(self, base_estimator="lr", n_bins=50, alpha_grid_size=100):
        self.base_estimator = base_estimator
        self.n_bins = n_bins
        self.alpha_grid_size = alpha_grid_size

        self.alpha_hat_ = None
        self.alphas_ = None
        self.violation_curve_ = None
        self.s_p_ = None
        self.s_u_ = None

    def estimate(self, X, y):
        X_u = X[np.where(y == 0)[0], :]
        X_p = X[np.where(y == 1)[0], :]
        self.s_p_, self.s_u_ = self._get_calibrated_scores(X_p, X_u)
        bins = np.linspace(0, 1, self.n_bins + 1)

        p_hist, _ = np.histogram(self.s_p_, bins=bins, density=True)
        u_hist, _ = np.histogram(self.s_u_, bins=bins, density=True)

        p_hist += 1e-10
        u_hist += 1e-10
        alphas = np.linspace(0.01, 0.99, self.alpha_grid_size)
        violations = []

        for alpha in alphas:
            diff = u_hist - alpha * p_hist
            negative_mass = np.sum(np.abs(diff[diff < 0]))

            bin_width = 1.0 / self.n_bins
            total_violation = negative_mass * bin_width
            violations.append(total_violation)

        self.alphas_ = alphas
        self.violation_curve_ = np.array(violations)
        self.alpha_hat_ = self._find_elbow(self.violation_curve_)
        return {"alpha": self.alpha_hat_}

    def _get_calibrated_scores(self, X_p, X_u):
        """
        Uses CalibratedClassifierCV to ensure scores are actual probabilities.
        This prevents the "clumping" that confuses AlphaMax.
        """
        X = np.vstack((X_u, X_p))
        y = np.hstack((np.zeros(len(X_u)), np.ones(len(X_p))))

        # Base estimator: Logistic Regression is safer than MLP for limited data
        if self.base_estimator == "lr":
            base = LogisticRegression(C=1.0)
        elif self.base_estimator == "mlp":
            base = MLPClassifier(
                activation="relu",
                solver="adam",
            )
        elif self.base_estimator == "dt":
            base = DecisionTreeClassifier()
        else:
            raise ValueError(f"Estimator {self.base_estimator} not known")

        # Calibration wrapper
        # We use 'isotonic' if we have enough data, else 'sigmoid'
        calibration_method = "sigmoid" if len(X) < 1000 else "isotonic"
        calibrated_clf = CalibratedClassifierCV(base, method=calibration_method, cv=5)

        # We need "clean" scores. We can't train and predict on the same data.
        # We perform a manual K-Fold for prediction.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        probs = np.zeros(len(y))

        scaler = StandardScaler()

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            # Scale
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            calibrated_clf.fit(X_train_s, y_train)
            probs[test_idx] = calibrated_clf.predict_proba(X_test_s)[:, 1]

        return probs[len(X_u) :], probs[: len(X_u)]

    def _find_elbow(self, y):
        """
        Finds the point of maximum curvature on the violation curve.
        Ideally, violation is 0 until alpha_true, then rises.
        """
        # We look for the point where the slope increases most dramatically
        # (Maximum second derivative)
        d1 = np.gradient(y)
        d2 = np.gradient(d1)

        # Find index of max positive curvature (start of the ramp)
        # We skip the very end (alpha > 0.9) as it's unstable
        search_limit = int(0.9 * len(y))
        idx = np.argmax(d2[:search_limit])

        return self.alphas_[idx]

    def plot(self):
        plt.figure(figsize=(12, 5))

        # Plot 1: Score Densities
        plt.subplot(1, 2, 1)
        plt.hist(
            self.s_p_,
            bins=self.n_bins,
            alpha=0.5,
            label="P",
            density=True,
            color="blue",
        )
        plt.hist(
            self.s_u_,
            bins=self.n_bins,
            alpha=0.5,
            label="U",
            density=True,
            color="grey",
        )
        plt.title("Calibrated Score Densities")
        plt.legend()

        # Plot 2: Violation Curve
        plt.subplot(1, 2, 2)
        plt.plot(
            self.alphas_, self.violation_curve_, "k-", label="Constraint Violation"
        )
        plt.axvline(
            self.alpha_hat_,
            color="r",
            linestyle="--",
            label=f"Est: {self.alpha_hat_:.3f}",
        )
        plt.title("Alpha Estimation (Look for the 'Corner')")
        plt.xlabel("Alpha")
        plt.ylabel("Negative Mass Violation")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()
