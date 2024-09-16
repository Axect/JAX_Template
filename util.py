import jax
import jax.numpy as jnp
import equinox as eqx
import wandb
import survey
import polars as pl
import numpy as np
import random


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


def load_data():
    pass


def set_seed(seed: int, n_items: int):
    np.random.seed(seed)
    random.seed(seed)
    return jax.random.split(jax.random.PRNGKey(seed), n_items)


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device="cpu"):
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
        return loss, model

    @eqx.filter_jit
    def val_epoch(self, model, dl_val):
        total_loss = 0.0
        for x, y in dl_val:
            loss = self.criterion(model, x, y)
            total_loss += loss
        return total_loss

    def train(self, dl_train, dl_val, epochs):
        model = self.model
        opt_state = self.optim.init(model)
        val_loss = 0.0
        for epoch in range(epochs):
            opt_state.hyperparams['learning_rate'] = self.scheduler(epoch)
            
            train_loss, model = train_epoch(model, dl_train, opt_state)
            val_loss = val_epoch(model, dl_val)

            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "lr": self.scheduler(epoch)})

        return val_loss
