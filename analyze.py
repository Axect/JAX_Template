import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from util import Trainer, DataLoader, select_project, select_group, select_seed, select_device, load_model, load_data, load_study, load_best_model


class Tester(Trainer):
    def __init__(self, model):
        super().__init__(model, None, None)


def main():
    project = select_project()
    group_name = select_group(project)
    seed = select_seed(project, group_name)
    device = select_device()
    jax.config.update("jax_default_device", device)

    model, config = load_model(project, group_name, seed)

    test_key = jax.random.PRNGKey(1)
    test_dataset = load_data(1000, test_key)
    test_loader = DataLoader(test_dataset, config.batch_size, shuffle=False, drop_last=False)

    tester = Tester(model)
    test_loss = tester.val_epoch(model, test_loader)

    print(f"Test loss: {test_loss:.4e}")


if __name__ == "__main__":
    main()
