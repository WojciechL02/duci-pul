import numpy as np
import scipy.stats as st

# ---------- PU metrics and conversion ----------
def tpr_pu_at_threshold(scores_valid, S_valid, thr):
    """TPR_PU(thr) = sum S_i * 1[score>thr] / sum S_i over (N+U)."""
    m_hat = (scores_valid > thr).astype(float)
    denom = float(S_valid.sum())
    if denom == 0:
        return 0.0
    value = float((S_valid * m_hat).sum() / denom)
    return np.clip(value, 0.0, 1.0)

def fpr_at_threshold(neg_scores_valid, thr):
    """FPR(thr) on known negatives only."""
    return float((neg_scores_valid > thr).mean()) if len(neg_scores_valid) else 0.0

def tpr_nu_at_threshold(scores_valid, thr):
    """TPR_NU(thr) = mean(1[score>thr]) over all samples when only negative samples are available."""
    return float((scores_valid > thr).mean()) if len(scores_valid) else 0.0

def fpr_nu_at_threshold(neg_scores_valid, thr):
    """FPR_NU(thr) on known negatives only. Same as fpr_at_threshold."""
    return fpr_at_threshold(neg_scores_valid, thr)

def convert_tpr_pu_to_true(TPR_PU, FPR, pi_hat, S_P_hat, S_N_hat):
    """
    From soft-label PU: TPR_PU = a*TPR + b*FPR, with
      a = (pi*S_P) / (pi*S_P + (1-pi)*S_N)
      b = ((1-pi)*S_N) / (pi*S_P + (1-pi)*S_N)
    => TPR = (TPR_PU - b*FPR) / a
    """
    denom = (pi_hat * S_P_hat + (1 - pi_hat) * S_N_hat)
    if denom <= 1e-12:
        raise ValueError("Denominator for (a,b) ~ 0; check pi_hat/S_P_hat/S_N_hat.")
    a = (pi_hat * S_P_hat) / denom
    b = ((1 - pi_hat) * S_N_hat) / denom
    if abs(a) <= 1e-12:
        raise ValueError("Coefficient 'a' ~ 0; cannot invert TPR_PU -> TPR.")
    TPR = (TPR_PU - b * FPR) / a
    return float(TPR), float(a), float(b)

def convert_tpr_nu_to_true(TPR_NU, FPR, pi_hat):
    """
    When only negative samples are available: TPR_NU = pi*TPR + (1-pi)*FPR
    => TPR = (TPR_NU - (1-pi)*FPR) / pi
    """
    if pi_hat <= 1e-12:
        raise ValueError("pi_hat ~ 0; cannot invert TPR_NU -> TPR.")
    TPR = (TPR_NU - (1 - pi_hat) * FPR) / pi_hat
    return float(TPR)

def estimate_class_prior(scores, neg_scores, simple_mean=True):
    """
    Estimate the class prior (pi) using both KM method and simple mean.

    Args:
        scores: Scores of all samples
        neg_scores: Scores of known negative samples

    Returns:
        pi_hat: Estimated class prior using KM method
    """
    # Simple mean estimation
    mean_estimate = np.mean(scores)
    print(f"Simple mean estimate of pi: {mean_estimate:.4f}")

    if simple_mean:
        return float(np.clip(mean_estimate, 0.0, 1.0))
    # Original KM method implementation
    # Sort scores in ascending order
    sorted_scores = np.sort(scores)
    sorted_neg_scores = np.sort(neg_scores)

    # Calculate empirical CDFs
    cdf_all = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    cdf_neg = np.arange(1, len(sorted_neg_scores) + 1) / len(sorted_neg_scores)

    # Interpolate negative CDF to match all scores
    from scipy.interpolate import interp1d
    f_neg = interp1d(sorted_neg_scores, cdf_neg, bounds_error=False, fill_value=(0, 1))
    neg_cdf_interp = f_neg(sorted_scores)

    # Calculate pi_hat as the maximum difference between CDFs
    valid_indices = (neg_cdf_interp > 0) & (cdf_all > 0)
    if not np.any(valid_indices):
        return 0.0

    ratios = cdf_all[valid_indices] / neg_cdf_interp[valid_indices]
    pi_hat = 1.0 - np.min(ratios)

    print(f"KM method estimate of pi: {pi_hat:.4f}")
    return float(np.clip(pi_hat, 0.0, 1.0))


def choose_threshold(scores_valid, neg_scores_valid, S_valid, pi_hat, S_P_hat, S_N_hat):
    """Sweep thresholds; pick argmax of J = TPR - FPR."""
    thresholds = np.unique(scores_valid)
    if len(thresholds) > 500:
        thresholds = np.quantile(scores_valid, np.linspace(0, 1, 501))

    best = {"thr": None, "J": -np.inf, "TPR": None, "FPR": None, "a": None, "b": None, "TPR_PU": None}
    for thr in thresholds:
        FPR = fpr_at_threshold(neg_scores_valid, thr)
        TPR_PU = tpr_pu_at_threshold(scores_valid, S_valid, thr)
        TPR, a, b = convert_tpr_pu_to_true(TPR_PU, FPR, pi_hat, S_P_hat, S_N_hat)
        J = TPR - FPR
        if J > best["J"]:
            best = {"thr": float(thr), "J": float(J), "TPR": float(np.clip(TPR, 0.0, 1.0)), "FPR": float(FPR),
                    "a": a,
                    "b": b,
                    "TPR_PU": float(TPR_PU)
                    }
    return best

def choose_threshold_nu(scores_valid, neg_scores_valid, pi_hat):
    """Sweep thresholds with only negative samples; pick argmax of J = TPR - FPR."""
    thresholds = np.unique(scores_valid)
    if len(thresholds) > 500:
        thresholds = np.quantile(scores_valid, np.linspace(0, 1, 501))

    best = {"thr": None, "J": -np.inf, "TPR": None, "FPR": None, "TPR_NU": None}
    for thr in thresholds:
        FPR = fpr_nu_at_threshold(neg_scores_valid, thr)
        TPR_NU = tpr_nu_at_threshold(scores_valid, thr)
        TPR = convert_tpr_nu_to_true(TPR_NU, FPR, pi_hat)
        J = TPR - FPR
        if J > best["J"]:
            best = {"thr": float(thr), "J": float(J), "TPR": float(np.clip(TPR, 0.0, 1.0)), "FPR": float(FPR),
                    "TPR_NU": float(TPR_NU)
                    }
    return best

# ---------- DUCI debiasing ----------
def debias_target(target_scores, thr, TPR, FPR, alpha=0.05):
    """DUCI Eq. (4) debiasing on target set with CLT CI."""
    if abs(TPR - FPR) <= 1e-12:
        raise ValueError("TPR and FPR too close; unstable debiasing.")
    m_hat = (target_scores > thr).astype(float)
    p_i_hat = (m_hat - FPR) / (TPR - FPR)
    p_hat = float(p_i_hat.mean())
    s2 = p_i_hat.var(ddof=1) if len(p_i_hat) > 1 else 0.0
    z = st.norm.ppf(0.975)
    half = z * np.sqrt(s2 / len(p_i_hat)) if len(p_i_hat) > 0 else 0.0
    return {
        "threshold": float(thr),
        "TPR": float(TPR),
        "FPR": float(FPR),
        "J": float(TPR - FPR),
        "p_hat": float(p_hat),
        "ci_low": float(p_hat - half),
        "ci_high": float(p_hat + half),
    }

def estimate_p_and_debias(target_scores, neg_scores, alpha=0.05):
    """Estimate p (class prior) and perform debiasing with only negative samples."""
    # Estimate class prior
    pi_hat = estimate_class_prior(target_scores, neg_scores)
    print(f"Estimated pi: {pi_hat:.4f}")
    # Choose optimal threshold
    best = choose_threshold_nu(target_scores, neg_scores, pi_hat)

    # Perform debiasing
    result = debias_target(target_scores, best["thr"], best["TPR"], best["FPR"], alpha)

    # Add pi_hat to the result
    result["pi_hat"] = float(pi_hat)

    # Calculate lowerbound (p_hat with TPR=1.0)
    # Find threshold that minimizes FPR (since TPR is fixed at 1.0)
    thresholds = np.unique(target_scores)
    if len(thresholds) > 500:
        thresholds = np.quantile(target_scores, np.linspace(0, 1, 501))

    best_thr = None
    best_fpr = float('inf')

    for thr in thresholds:
        fpr = fpr_nu_at_threshold(neg_scores, thr)
        if fpr < best_fpr:
            best_fpr = fpr
            best_thr = thr

    # Set TPR to 1.0 for lowerbound calculation
    tpr = 1.0

    # Perform debiasing with TPR=1.0
    lowerbound_result = debias_target(target_scores, best_thr, tpr, best_fpr, alpha)

    # Add lowerbound p_hat to the main result
    result["lowerbound"] = float(lowerbound_result["p_hat"])

    return result

