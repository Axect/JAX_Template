import jax
import jax.numpy as jnp
import equinox as eqx
import wandb
import numpy as np
import survey
import optuna

import random
import os
import math
from typing import Tuple, Iterator

from config import RunConfig, OptimizeConfig


def save(model, path):
    with open(path, 'wb') as f:
        eqx.tree_serialise_leaves(f, model)


def load(model, path):
    with open(path, 'rb') as f:
        return eqx.tree_deserialise_leaves(f, model)


class Dataset:
    def __init__(self, data: Tuple[jnp.ndarray, ...]):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.data)


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.num_samples = len(dataset)
        self.num_batches = self.num_samples // batch_size
        if not drop_last and self.num_samples % batch_size != 0:
            self.num_batches += 1

    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, ...]]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_samples)
            batch_indices = indices[start:end]
            yield self.dataset[batch_indices]

    def __len__(self):
        return self.num_batches


def load_data(dataset_size: int, key: jnp.ndarray) -> Dataset:
    x = jnp.linspace(0, 2 * jnp.pi, dataset_size)
    x = x.reshape((dataset_size, 1))
    epsilon = jax.random.normal(key, x.shape) * 0.01
    y = jnp.sin(x) + epsilon
    return Dataset((x, y))


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    return jax.random.PRNGKey(seed)


class Trainer:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        if scheduler is not None:
            self.scheduler = scheduler.gen_scheduler()
        self.optim = optimizer

    @eqx.filter_jit
    def step(self, model, x):
        return jax.vmap(model)(x)
    
    @eqx.filter_jit
    def loss_fn(self, model, x, y):
        y_pred = self.step(model, x)
        return jnp.mean((y - y_pred) ** 2)

    @eqx.filter_jit
    def train_step(self, model, x, y):
        loss, grads = eqx.filter_value_and_grad(self.loss_fn)(model, x, y)
        return loss, grads

    @eqx.filter_jit
    def update_params(self, grads, opt_state, model):
        updates, opt_state = self.optim.update(grads, opt_state, params=model)
        return updates, opt_state

    @eqx.filter_jit
    def val_step(self, model, x, y):
        return self.loss_fn(model, x, y)

    #@eqx.filter_jit # If model is lightweight, use this
    def train_epoch(self, model, dl_train, opt_state):
        total_loss = 0.0
        for batch in dl_train:
            x, y = batch
            loss, grads = self.train_step(model, x, y)
            updates, opt_state = self.update_params(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            total_loss += loss
        loss = total_loss / len(dl_train)
        return loss, model

    #@eqx.filter_jit # If model is lightweight, use this
    def val_epoch(self, model, dl_val):
        total_loss = 0.0
        for batch in dl_val:
            x, y = batch
            loss = self.val_step(model, x, y)
            total_loss += loss
        return total_loss / len(dl_val)

    def train(self, dl_train, dl_val, epochs):
        model = self.model
        opt_state = self.optim.init(model)
        val_loss = 0.0
        for epoch in range(epochs):
            opt_state.hyperparams['learning_rate'] = self.scheduler(epoch)
            
            train_loss, model = self.train_epoch(model, dl_train, opt_state)
            val_loss = self.val_epoch(model, dl_val)

            # Early stopping if loss becomes NaN
            if math.isnan(train_loss) or math.isnan(val_loss):
                print("Early stopping due to NaN loss")
                val_loss = math.inf
                break

            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "lr": self.scheduler(epoch)})
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, train_loss: {train_loss:.4e}, val_loss: {val_loss:.4e}")

        return val_loss


def run(run_config: RunConfig, train_dataset: Dataset, val_dataset: Dataset, group_name = None) -> float:
    project = run_config.project
    seeds = run_config.seeds
    if not group_name:
        group_name = run_config.gen_group_name()
    tags = run_config.gen_tags()

    group_path = f"runs/{run_config.project}/{group_name}"
    if not os.path.exists(group_path):
        os.makedirs(group_path)
    run_config.to_yaml(f"{group_path}/config.yaml")

    total_loss = 0
    for seed in seeds:
        model_key = set_seed(seed)

        model = run_config.create_model(model_key)
        optimizer = run_config.create_optimizer()
        scheduler = run_config.create_scheduler()

        dl_train = DataLoader(train_dataset, run_config.batch_size)
        dl_val = DataLoader(val_dataset, run_config.batch_size)

        run_name = f"{seed}"
        wandb.init(
            project=project,
            name=run_name,
            group=group_name,
            tags=tags,
            config=run_config.gen_config(),
        )

        trainer = Trainer(model, optimizer, scheduler)
        val_loss = trainer.train(dl_train, dl_val, run_config.epochs)
        total_loss += val_loss

        # Save model
        run_path = f"{group_path}/{run_name}"
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        save(model, f"{run_path}/model.eqx")

        wandb.finish()

    return total_loss / len(seeds)


# ┌──────────────────────────────────────────────────────────┐
#  For Analyze
# └──────────────────────────────────────────────────────────┘
def select_project():
    runs_path = "runs/"
    projects = [d for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d))]
    if not projects:
        raise ValueError(f"No projects found in {runs_path}")
    
    selected_index = survey.routines.select("Select a project:", options=projects)
    return projects[selected_index] # pyright: ignore


def select_group(project):
    runs_path = f"runs/{project}"
    groups = [d for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d))]
    if not groups:
        raise ValueError(f"No run groups found in {runs_path}")
    
    selected_index = survey.routines.select("Select a run group:", options=groups)
    return groups[selected_index] # pyright: ignore

def select_seed(project, group_name):
    group_path = f"runs/{project}/{group_name}"
    seeds = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]
    if not seeds:
        raise ValueError(f"No seeds found in {group_path}")
    
    selected_index = survey.routines.select("Select a seed:", options=seeds)
    return seeds[selected_index] # pyright: ignore

def select_device():
    devices = list(jax.devices())
    devices_str = [f"{d}" for d in devices]
    selected_index = survey.routines.select("Select a device:", options=devices_str)
    return devices[selected_index] # pyright: ignore


def load_model(project, group_name, seed):
    """
    Load a trained model and its configuration.

    Args:
        project (str): The name of the project.
        group_name (str): The name of the run group.
        seed (str): The seed of the specific run.

    Returns:
        tuple: A tuple containing the loaded model and its configuration.

    Raises:
        FileNotFoundError: If the config or model file is not found.

    Example usage:
        model, config = load_model("MyProject", "experiment1", "seed42")
    """
    config_path = f"runs/{project}/{group_name}/config.yaml"
    model_path = f"runs/{project}/{group_name}/{seed}/model.eqx"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found for {project}/{group_name}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for {project}/{group_name}/{seed}")
    
    config = RunConfig.from_yaml(config_path)
    model_key = jax.random.PRNGKey(int(seed))
    model = config.create_model(model_key)
    model = load(model, model_path)
    
    return model, config


def load_study(project, study_name):
    """
    Load the best study from an optimization run.

    Args:
        project (str): The name of the project.
        study_name (str): The name of the study.

    Returns:
        optuna.Study: The loaded study object.
    """
    study = optuna.load_study(
        study_name=study_name,
        storage=f'sqlite:///{project}.db'
    )
    return study


def load_best_model(project, study_name):
    """
    Load the best model and its configuration from an optimization study.

    Args:
        project (str): The name of the project.
        study_name (str): The name of the study.

    Returns:
        tuple: A tuple containing the loaded model, its configuration, and the best trial number.
    """
    study = load_study(project, study_name)
    best_trial = study.best_trial
    project_name = project
    group_name = best_trial.user_attrs['group_name']

    # Select Seed
    seed = select_seed(project_name, group_name)
    best_model, best_config = load_model(project_name, group_name, seed)

    return best_model, best_config
