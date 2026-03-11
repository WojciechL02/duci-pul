from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


def load_npz(path):
    data = np.load(
        path,
        allow_pickle=True,
    )
    if np.any(np.isnan(data["data"])):
        raise ValueError(f"NaNs in data {path}")
    return data["data"]


def load_files(prefix: str, data_type: str, data_dir: str):
    folder = Path(data_dir)
    mem_pattern = f"{prefix}_*{data_type}*_mem*.npz"
    nonmem_pattern = f"{prefix}_*{data_type}*_nonmem*.npz"
    if data_type == "from":
        mem_pattern = f"{prefix}_*{data_type}-mem*.npz"
        nonmem_pattern = f"{prefix}_*{data_type}-nonmem*.npz"
    mem_matches = list(folder.glob(mem_pattern))
    nonmem_matches = list(folder.glob(nonmem_pattern))
    if len(mem_matches) > 1 or len(nonmem_matches) > 1:
        raise ValueError(f"More than one file for {prefix}-{data_type} found!")
    return mem_matches[0], nonmem_matches[0]


def load_data(name: str, data_type: str, data_dir: str):
    mem_match, nonmem_match = load_files(name, data_type, data_dir)
    members = load_npz(mem_match)
    members = np.unique(members, axis=0)
    nonmembers = load_npz(nonmem_match)
    nonmembers = np.unique(nonmembers, axis=0)
    return members, nonmembers


def permute_datasets(permutation, datasets):
    permuted = []
    for dataset in datasets:
        permuted.append(dataset[permutation[: len(dataset)]])
    return permuted


def prepare_data(
    name, seed, p, ss_len=2000, run_type="real", data_dir="data", num_real_negatives=0, bootstrap=False
):
    """
    Prepare data for the experiment.

    Parameters
    ----------
    name : str
        The name of the dataset.
    seed : int
        The random seed.
    p : float
        Target member prevalence in the *test unlabeled* set (0..1)
    ss_len : int
        Size of the suspect set.
    run_type : str
        Type of the experiment: "real", "ae_synth", "correction".
    data_dir: str
        Data source directory.
    num_real_negatives: int
        How many known negatives to use. If 0 then synthetic will be used. Works only in ''real'' run type.
    bootstrap : bool
        Whether to sample data with replacement (each run should have different seed then).

    Returns
    -------
    X_test  : (n_test, d) float32
    y_test  : (n_test,)  int   # true labels: members=0, nonmembers=1
    s_test  : (n_test,)  int   # observed NU labels for test (here: all 0 = unlabeled)
    X_ctrl_test : (n_test, d) float32  # for correction only
    """
    np.random.seed(seed)

    if run_type == "real":
        X_mem, X_nonmem = load_data(name, "real", data_dir)
        X_mem_generated = X_nonmem.copy()
        X_nonmem_generated = X_nonmem.copy()

    elif run_type == "synth":
        X_mem, X_nonmem = load_data(name, "real", data_dir)
        X_mem_generated, X_nonmem_generated = load_data(name, "from", data_dir)
    elif run_type in ("ae_synth", "correction"):
        X_mem, X_nonmem = load_data(name, "ae", data_dir)
        X_mem_generated, X_nonmem_generated = load_data(name, "from", data_dir)
    else:
        raise FileNotFoundError(f"No matching files found for run_type {run_type}.")

    N = min(len(X_mem), len(X_nonmem), len(X_mem_generated), len(X_nonmem_generated))
    perm = np.random.permutation(N)
    X_mem, X_nonmem, X_mem_generated, X_nonmem_generated = permute_datasets(
        perm, [X_mem, X_nonmem, X_mem_generated, X_nonmem_generated]
    )

    if bootstrap:
        boot_idx = np.random.choice(N, size=N, replace=True)
        X_mem = X_mem[boot_idx]
        X_nonmem = X_nonmem[boot_idx]
        X_mem_generated = X_mem_generated[boot_idx]
        X_nonmem_generated = X_nonmem_generated[boot_idx]

    n_pos_test = int(ss_len * p)  # members inside suspect set
    n_unl_test_nonmem = ss_len - n_pos_test  # non-members inside test unlabeled

    if run_type == "real":
        X_test_nonmem = np.concatenate(
            [
                X_mem_generated[: int(ss_len / 2)],
                X_nonmem_generated[int(ss_len / 2) : ss_len],
            ]
        )
        if num_real_negatives > 0 and num_real_negatives < X_test_nonmem.shape[0]:
            X_test_nonmem = X_test_nonmem[np.random.randint(0, X_test_nonmem.shape[0], num_real_negatives)]
    else:
        X_test_nonmem = np.concatenate(
            [
                X_mem_generated[:n_pos_test],
                X_nonmem_generated[ss_len : ss_len + n_unl_test_nonmem],
            ]
        )

    X_test = np.concatenate(
        [
            X_test_nonmem,
            X_mem[:n_pos_test],
            X_nonmem[ss_len : ss_len + n_unl_test_nonmem],
        ]
    )

    y_test = np.concatenate(
        [
            np.ones(X_test_nonmem.shape[0], dtype=int),  # N-labeled
            np.zeros(n_pos_test, dtype=int),  # U-members
            np.ones(n_unl_test_nonmem, dtype=int),  # U-non-members
        ]
    )

    s_test = np.concatenate(
        [
            np.ones(X_test_nonmem.shape[0], dtype=int),  # N-labeled
            np.zeros(n_pos_test, dtype=int),  # U-members
            np.zeros(n_unl_test_nonmem, dtype=int),  # U-non-members
        ]
    )

    X_test = X_test.squeeze(1)
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)

    if run_type == "correction":
        control_name = (
            "ControlSEnsemble_coco" if "uvit" in name else "ControlSEnsemble_in"
        )
        mem, nonmem = load_files(control_name, "ae", data_dir)
        synth_mem, synth_nonmem = load_files(control_name, "from", data_dir)
        X_ctrl_ae_mem = load_npz(mem)
        X_ctrl_ae_nonmem = load_npz(nonmem)
        X_ctrl_mem_generated = load_npz(synth_mem)
        X_ctrl_nonmem_generated = load_npz(synth_nonmem)
        (
            X_ctrl_ae_mem,
            X_ctrl_ae_nonmem,
            X_ctrl_mem_generated,
            X_ctrl_nonmem_generated,
        ) = permute_datasets(
            perm,
            [
                X_ctrl_ae_mem,
                X_ctrl_ae_nonmem,
                X_ctrl_mem_generated,
                X_ctrl_nonmem_generated,
            ],
        )
        X_ctrl_test_nonmem = np.concatenate(
            [
                X_ctrl_mem_generated[:n_pos_test],
                X_ctrl_nonmem_generated[ss_len : ss_len + n_unl_test_nonmem],
            ]
        )

        X_ctrl_test = np.concatenate(
            [
                X_ctrl_test_nonmem,
                X_ctrl_ae_mem[:n_pos_test],
                X_ctrl_ae_nonmem[ss_len : ss_len + n_unl_test_nonmem],
            ]
        )
        X_ctrl_test = X_ctrl_test.squeeze(1)
        ols = LinearRegression().fit(X_ctrl_test, X_test)
        Xhat_test = ols.predict(X_ctrl_test)  # take only ctrl features explaining MIA
        scaler = MinMaxScaler()
        X_ctrl_test = scaler.fit_transform(Xhat_test)
        return X_test, y_test, s_test, X_ctrl_test

    return X_test, y_test, s_test, None
