import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import wandb
import survey
import optuna

from util import load_data, run
from config import RunConfig, OptimizeConfig

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--optimize_config", type=str, help="Path to the optimization YAML config file")
    args = parser.parse_args()

    # Run Config
    base_config = RunConfig.from_yaml(args.run_config)

    # Load data
    x_train, y_train = load_data(10000, jax.random.PRNGKey(0))
    x_val, y_val = load_data(2000, jax.random.PRNGKey(1))

    # Run
    if args.optimize_config:
        def objective(trial, base_config, optimize_config):
            params = optimize_config.suggest_params(trial)
            
            config = base_config.gen_config()
            config["project"] = f"{base_config.project}_Opt"
            for category, category_params in params.items():
                config[category].update(category_params)
            
            run_config = RunConfig(**config)
            group_name = run_config.gen_group_name()
            group_name += f"[{trial.number}]"

            trial.set_user_attr("group_name", group_name)

            return run(run_config, x_train, y_train, x_val, y_val, group_name)

        optimize_config = OptimizeConfig.from_yaml(args.optimize_config)
        study = optimize_config.create_study(project=f"{base_config.project}_Opt")
        study.optimize(lambda trial: objective(trial, base_config, optimize_config), n_trials=optimize_config.trials)

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print(f"  Path: runs/{base_config.project}_Opt/{trial.user_attrs['group_name']}")
        
    else:
        run(base_config, x_train, y_train, x_val, y_val)


if __name__ == "__main__":
    main()