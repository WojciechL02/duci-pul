import numpy as np
from sklearn.neighbors import KDTree
from scipy.special import digamma


def seed(seed):
    np.random.seed(seed)

class ThresholdOptimizer():
    def __init__(self, k, n) -> None:
        """
        k   The number of nearest neighbors to consider.
        n   The number of thresholds to consider.
        """
        self.k = k
        self.n = n

    def radius_nneighbor(self, Z, k, index):
        possible_radii = Z.squeeze()[max(0, index - k):min(Z.shape[0], index + k + 1)] - Z[index]
        possible_radii = np.sort(np.abs(possible_radii))
        return possible_radii[k]
    
    def lower_bound(self, T, k, tr):
        """
        tr  The relative threshold.
            This the fraction of the examples that would be on one side of the threshold: N=tr*T."
        """
        return digamma(T) - tr * digamma(tr * T) - (1 - tr) * digamma((1 - tr) * T) +  2 * k/T * digamma(k) - k/T * digamma(tr * T + k) - k/T * digamma((1 - tr) * T + k)

    def max_lower_bound(self, T, k):
        """
        Same as lower_bound(T,k,0.5) but slightly faster and 1 function call less.
        """
        return digamma(T) - digamma(T/2) + 2 * k/T * (digamma(k) - digamma(T/2 + k))

    def upper_bound(self, T, k, tr):
        """
        tr  The relative threshold.
            This the fraction of the examples that would be on one side of the threshold: N=tr*T."
        """
        return digamma(T) - tr * digamma(tr * T) - (1 - tr) * digamma((1 - tr) * T)

    def binary_search_sorted(self, fun, val, range_min, range_max, n_bins, precision):
        """
        Efficiently search for x so that fun(x)=val, for x in [range_min,range_max].
        precondition: fun(x) is increasing for x in [range_min,range_max].
        A binary search method is used
        """
        r = np.linspace(range_min, range_max, n_bins)
        vals = fun(r)
        i = np.searchsorted(vals,val)
        if (range_max - range_min)/n_bins < precision:
            return r[i-1]
        else:
            return self.binary_search_sorted(fun, val, r[i-1], r[i], n_bins, precision)
        
    def find_begin_range(self, T, k, n_bins=10, precision=1e-10):
        """
        Efficient solving of upper_bound(T,k,tr)=max_lower_bound(T,k) for tr in [0,0.5] 
        
        tr  is the relative threshold, 
            i.e. the fraction of the examples that would be on one side of the threshold: N=tr*T.
        """
        return self.binary_search_sorted(
            lambda tr: self.upper_bound(T, k, tr),
            self.max_lower_bound(T, k),
            range_min=precision, 
            range_max=0.5,
            n_bins=n_bins, 
            precision=precision
        )

    def get_threshold_range(self, T, k):
        """
        Returns the indices for the threshold range in Z
        """
        tr_min = self.find_begin_range(T, k)
        i_min = int(T * tr_min)
        i_max = T - i_min
        return i_min, i_max
    
    
    def find_threshold(self, Z):
        """
        Given a set of scores Z, finds the threshold that maximizes the mutual information.

        Z   The scores of the examples.
        """
        n_samples = Z.shape[0]

        Z = np.sort(Z)
        Z = Z.reshape(-1, 1)
        
        kd = KDTree(Z)

        # Find the bounds for the optimal thresholds.
        min_i, max_i = self.get_threshold_range(len(Z), self.k)
        # Select equally spaced thresholds to consider.
        threshold_candidates = np.linspace(start=Z[min_i], stop=Z[max_i], num=self.n)
        scores = np.zeros_like(threshold_candidates)

    
        for i, t in enumerate(threshold_candidates):
            y_tilde = np.where(Z > t, 1, 0)
            n_samples_pos = np.count_nonzero(y_tilde)
            n_samples_neg = n_samples - n_samples_pos
            enough_class_representation = n_samples_pos > self.k and n_samples_neg > self.k

            # If we can't calculate the score due to too few datapoints of a class, assign NaN.
            if not enough_class_representation:
                scores[i] = float('nan')
                continue
            
            label_counts_score = n_samples_neg/n_samples * digamma(n_samples_neg) + n_samples_pos/n_samples * digamma(n_samples_pos)
            nneighbours_counts = np.full((n_samples,), self.k)
            for index in range(n_samples_neg - self.k, n_samples_neg + self.k, 1):
                if y_tilde[index] == 0:
                    class_instances = Z[:n_samples_neg]
                    index_in_class = index
                else:
                    class_instances = Z[n_samples_neg:]
                    index_in_class = index - n_samples_neg
                r = self.radius_nneighbor(class_instances, self.k, index_in_class)

                nneighbours_counts[index] = kd.query_radius(Z[index].reshape(1, -1), r, count_only=True, return_distance=False)[0] - 1

            scores[i] = - np.mean(digamma(nneighbours_counts)) - label_counts_score
        
        return threshold_candidates[np.nanargmax(scores)]