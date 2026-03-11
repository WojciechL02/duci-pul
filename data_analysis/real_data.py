import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def analyze_full_dispersion(X_mem, X_nonmem, label_m="MEM", label_nm="NON-MEM"):
    # 1. Spłaszczenie danych do (n_samples, n_features)
    X_m = X_mem.reshape(X_mem.shape[0], -1)
    X_nm = X_nonmem.reshape(X_nonmem.shape[0], -1)

    # 2. Obliczenie centroidów obu klas
    centroid_m = np.mean(X_m, axis=0)
    centroid_nm = np.mean(X_nm, axis=0)

    # 3. Odległości wewnątrzklasowe (do własnego centroidu)
    dist_m_own = np.linalg.norm(X_m - centroid_m, axis=1)
    dist_nm_own = np.linalg.norm(X_nm - centroid_nm, axis=1)

    # 4. Odległości międzyklasowe (do PRZECIWNEGO centroidu)
    dist_m_opp = np.linalg.norm(X_m - centroid_nm, axis=1)
    dist_nm_opp = np.linalg.norm(X_nm - centroid_m, axis=1)

    # Statystyki pomocnicze
    r_m = np.mean(dist_m_own)
    r_nm = np.mean(dist_nm_own)
    r_m_opp = np.mean(dist_m_opp)
    r_nm_opp = np.mean(dist_nm_opp)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Wykres 1: Do własnego centroidu (Intra-class)
    axes[0].hist(
        dist_m_own,
        bins=50,
        alpha=0.5,
        label=f"{label_m} (do własnego)",
        color="blue",
        density=True,
    )
    axes[0].hist(
        dist_nm_own,
        bins=50,
        alpha=0.5,
        label=f"{label_nm} (do własnego)",
        color="red",
        density=True,
    )
    axes[0].axvline(r_m, color="blue", linestyle="dashed", linewidth=2)
    axes[0].axvline(r_nm, color="red", linestyle="dashed", linewidth=2)
    axes[0].set_title("Ściśnięcie: Odległość do własnego centroidu")
    axes[0].set_xlabel("Odległość Euklidesowa")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Wykres 2: Do przeciwnego centroidu (Inter-class)
    axes[1].hist(
        dist_m_opp,
        bins=50,
        alpha=0.5,
        label=f"{label_m} (do {label_nm})",
        color="cyan",
        density=True,
    )
    axes[1].hist(
        dist_nm_opp,
        bins=50,
        alpha=0.5,
        label=f"{label_nm} (do {label_m})",
        color="orange",
        density=True,
    )
    axes[1].axvline(r_m_opp, color="cyan", linestyle="dashed", linewidth=2)
    axes[1].axvline(r_nm_opp, color="orange", linestyle="dashed", linewidth=2)
    axes[1].set_title("Separacja: Odległość do przeciwnego centroidu")
    axes[1].set_xlabel("Odległość Euklidesowa")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_dispersion_stats(X_m, X_nm):
    # Ensure data is flattened
    X_m = X_m.reshape(X_m.shape[0], -1)
    X_nm = X_nm.reshape(X_nm.shape[0], -1)

    # Calculate Centroids
    centroid_m = np.mean(X_m, axis=0)
    centroid_nm = np.mean(X_nm, axis=0)

    # --- Helper function for printing in your requested format ---
    def print_case(data, center, label):
        dists = np.linalg.norm(data - center, axis=1)
        avg_dist = np.mean(dists)
        std_dist = np.std(dists)

        # Calculate Total Variance (Trace of Covariance Matrix)
        variances = np.var(data, axis=0, ddof=1)
        total_variance = np.sum(variances)

        print(f"=== Wyniki dla: {label} ===")
        print(f"Średnia odległość od środka (R_avg): {avg_dist:.6f} (±{std_dist:.4f})")
        print(f"Całkowita wariancja (Trace Σ):      {total_variance:.6f}")
        print("-" * 40)

    # 1. Intra-class Dispersion (Compression)
    print_case(X_m, centroid_m, "MEM (to own centroid)")
    print_case(X_nm, centroid_nm, "NON-MEM (to own centroid)")

    # 2. Inter-class Separation (Distance to Opposite)
    print_case(X_m, centroid_nm, "MEM (to NON-MEM centroid)")
    print_case(X_nm, centroid_m, "NON-MEM (to MEM centroid)")


model_name = "rar_xxl"
X_real_mem = np.load(
    f"../data/standard/{model_name}_llm_mia_cfg_real_mem_suspect_5k.npz"
)["data"].reshape(5000, -1)
X_real_nonmem = np.load(
    f"../data/standard/{model_name}_llm_mia_cfg_real_nonmem_suspect_5k.npz"
)["data"].reshape(5000, -1)


# Wywołanie dla Twoich danych
analyze_full_dispersion(X_real_mem, X_real_nonmem)
print_dispersion_stats(X_real_mem, X_real_nonmem)
