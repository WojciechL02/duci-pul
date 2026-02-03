import time
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import prepare_data
from src.utils import save_results, print_summary, seed_everything
from src.methods.pul_based import PULBased
from src.methods.mpe_based import MPEBased


MPE_METHODS = ["km", "dedpul", "tice", "alphamax"]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print(100 * "=")
    print(
        f"Run type={cfg.run_type} | Method={cfg.method.name} | Data={cfg.target} | p={cfg.prob} | ss_len={cfg.ss_len}"
    )
    print(100 * "=")
    print("\nConfig:")
    print(OmegaConf.to_yaml(cfg))
    print(100 * "-")

    tstart = time.time()
    records = []
    for run in range(cfg.n_runs):
        seed = cfg.seed * 1000 + int(run)
        seed_everything(seed)
        X_test, y_test, s_test, X_ctrl_test = prepare_data(
            name=cfg.target,
            seed=seed,
            p=cfg.prob,
            run_type=cfg.run_type,
            ss_len=cfg.ss_len,
            data_dir=cfg.data_dir,
            bootstrap=cfg.bootstrap,
        )

        if cfg.method.name in MPE_METHODS:
            estimator = MPEBased(
                cfg.method.name, cfg.method.mpe_args, run_type=cfg.run_type, seed=seed
            )
        else:
            estimator = PULBased(
                cfg.method.name,
                cfg.method.pul_args,
                cfg.method.method_args,
                cfg.run_type,
                seed,
            )
        results = estimator.fit_estimate(X_test, y_test, s_test, X_ctrl_test)
        print(f"Run {run + 1}/{cfg.n_runs}: p_hat={results['p_hat_test']:.3f}")
        records.append(results)

    metrics = [
        "p_hat_test",
    ]
    if cfg.run_type == "correction":
        metrics += ["p_hat_ctrl", "p_hat_comb", "p_hat_2MIA-comb", "p_hat_MIA-ctrl"]

    df = save_results(records, metrics, cfg.results_dir, cfg)
    print_summary(df, cfg)
    print("[Elapsed time = {:.1f} s]".format(time.time() - tstart))
    print("Done!")


if __name__ == "__main__":
    main()
