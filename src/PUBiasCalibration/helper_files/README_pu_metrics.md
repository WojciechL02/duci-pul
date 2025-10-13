# PU Metrics and Debiasing

This module provides functions for estimating TPR, FPR, and performing debiasing in Positive-Unlabeled (PU) learning scenarios, particularly when only negative samples are available.

## Functions

### TPR and FPR Estimation

- `tpr_pu_at_threshold(scores_valid, S_valid, thr)`: Calculates TPR_PU at a given threshold using labeled samples.
- `fpr_at_threshold(neg_scores_valid, thr)`: Calculates FPR at a given threshold using known negative samples.
- `tpr_nu_at_threshold(scores_valid, thr)`: Calculates TPR_NU at a given threshold when only negative samples are available.
- `fpr_nu_at_threshold(neg_scores_valid, thr)`: Same as `fpr_at_threshold`, calculates FPR at a given threshold.

### Conversion Functions

- `convert_tpr_pu_to_true(TPR_PU, FPR, pi_hat, S_P_hat, S_N_hat)`: Converts TPR_PU to true TPR using the formula TPR = (TPR_PU - b*FPR) / a.
- `convert_tpr_nu_to_true(TPR_NU, FPR, pi_hat)`: Converts TPR_NU to true TPR when only negative samples are available.

### Class Prior Estimation

- `estimate_class_prior(scores, neg_scores)`: Estimates the class prior (pi) using the empirical CDF method.

### Threshold Selection

- `choose_threshold(scores_valid, neg_scores_valid, S_valid, pi_hat, S_P_hat, S_N_hat)`: Selects the optimal threshold that maximizes J = TPR - FPR in PU learning.
- `choose_threshold_nu(scores_valid, neg_scores_valid, pi_hat)`: Selects the optimal threshold when only negative samples are available.

### Debiasing

- `debias_target(target_scores, thr, TPR, FPR, alpha=0.05)`: Performs DUCI debiasing on target set with confidence intervals.
- `estimate_p_and_debias(target_scores, neg_scores, alpha=0.05)`: Combines class prior estimation and debiasing in one function.

## Usage Example

```python
import numpy as np
from PUBiasCalibration.helper_files.pu_metrics import (
    estimate_class_prior,
    choose_threshold_nu,
    estimate_p_and_debias
)

# Example with only negative samples available
# scores: scores of all samples
# neg_scores: scores of known negative samples

# Estimate class prior
pi_hat = estimate_class_prior(scores, neg_scores)

# Choose optimal threshold
best = choose_threshold_nu(scores, neg_scores, pi_hat)

# Perform debiasing
result = estimate_p_and_debias(scores, neg_scores)

# Access results
print(f"Estimated class prior: {pi_hat:.4f}")
print(f"Optimal threshold: {best['thr']:.4f}")
print(f"Estimated TPR: {best['TPR']:.4f}")
print(f"Estimated FPR: {best['FPR']:.4f}")
print(f"Estimated prevalence: {result['p_hat']:.4f}")
```

## Notes

- When using `convert_tpr_pu_to_true`, ensure that pi_hat, S_P_hat, and S_N_hat are properly estimated to avoid numerical issues.
- When using `convert_tpr_nu_to_true`, ensure that pi_hat is not too close to zero to avoid numerical instability.
- The `estimate_class_prior` function uses a simple empirical CDF method, which may not be optimal for all datasets. Consider using more sophisticated methods for better estimation.