import jax
import jax.numpy as jnp
import equinox as eqx
import wandb
import survey
import polars as pl
import numpy as np

import random
import os
import math

from config import RunConfig, OptimizeConfig


def save(model, path):
    with open(path, 'wb') as f:
        eqx.tree_serialise_leaves(f, model)


class DataLoader:
    def __init__(self, x, y, batch_size, shuffle=True, drop_last=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.num_samples = x.shape[0]
        self.num_batches = self.num_samples // batch_size
        if not drop_last and self.num_samples % batch_size != 0:
            self.num_batches += 1

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
            self.x = self.x[indices]
            self.y = self.y[indices]
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_samples)
            yield self.x[start:end], self.y[start:end]

    def __len__(self):
        return self.num_batches


def load_data(dataset_size, key):
    x = jnp.linspace(0, 2 * jnp.pi, dataset_size)
    x = x.reshape((dataset_size, 1))
    epsilon = jax.random.normal(key, x.shape) * 0.01
    y = jnp.sin(x) + epsilon
    return x, y


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    return jax.random.PRNGKey(seed)


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion):
        self.model = model
        self.scheduler = scheduler.gen_scheduler()
        self.optim = optimizer
        self.criterion = criterion

    @eqx.filter_jit
    def train_epoch(self, model, dl_train, opt_state):
        total_loss = 0.0
        for x, y in dl_train:
            loss, grads = self.criterion(model, x, y)
            updates, _ = self.optim.update(grads, opt_state, params=model)
            model = eqx.apply_updates(model, updates)
            total_loss += loss
        loss = total_loss / len(dl_train)
        return loss, model

    @eqx.filter_jit
    def val_epoch(self, model, dl_val):
        total_loss = 0.0
        for x, y in dl_val:
            loss, _ = self.criterion(model, x, y)
            total_loss += loss
        return total_loss

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


def run(run_config: RunConfig, x_train, y_train, x_val, y_val, group_name=None):
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

        dl_train = DataLoader(x_train, y_train, run_config.batch_size)
        dl_val = DataLoader(x_val, y_val, run_config.batch_size)

        run_name = f"{seed}"
        wandb.init(
            project=project,
            name=run_name,
            group=group_name,
            tags=tags,
            config=run_config.gen_config(),
        )

        trainer = Trainer(model, optimizer, scheduler, mse)
        val_loss = trainer.train(dl_train, dl_val, run_config.epochs)
        total_loss += val_loss

        # Save model
        run_path = f"{group_path}/{run_name}"
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        save(model, f"{run_path}/model.eqx")

        wandb.finish()

    return total_loss / len(seeds)


@eqx.filter_value_and_grad
def mse(model, x, y):
    y_hat = jax.vmap(model)(x)
    return jnp.mean((y - y_hat) ** 2)
