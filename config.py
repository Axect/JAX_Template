from dataclasses import dataclass, asdict, field
import yaml
import importlib

import jax
import optax
import optuna


@dataclass
class RunConfig:
    project: str
    device: str
    seeds: list[int]
    net: str
    optimizer: str
    scheduler: str
    epochs: int
    batch_size: int
    net_config: dict[str, int]
    optimizer_config: dict[str, int | float]
    scheduler_config: dict[str, int | float]

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return cls(**config)

    def to_yaml(self, path: str):
        with open(path, 'w') as file:
            yaml.dump(asdict(self), file, sort_keys=False)

    def create_model(self, key):
        module_name, class_name = self.net.rsplit('.', 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class(self.net_config, key=key)

    def create_optimizer(self):
        module_name, class_name = self.optimizer.rsplit('.', 1)
        module = importlib.import_module(module_name)
        optimizer_class = getattr(module, class_name)
        return optax.inject_hyperparams(optimizer_class)(**self.optimizer_config)

    def create_scheduler(self):
        scheduler_module, scheduler_class = self.scheduler.rsplit('.', 1)
        scheduler_module = importlib.import_module(scheduler_module)
        scheduler_class = getattr(scheduler_module, scheduler_class)
        return scheduler_class(**self.scheduler_config)

    def gen_group_name(self):
        name = f"{self.net.split('.')[-1]}"
        for k, v in self.net_config.items():
            name += f"_{k[0]}_{v}"
        name += f"_{abbreviate(self.optimizer.split('.')[-1])}"
        for k, v in self.optimizer_config.items():
            name += f"_{k[0]}_{v:.4e}" if isinstance(v, float) else f"_{k[0]}_{v}"
        name += f"_{abbreviate(self.scheduler.split('.')[-1])}"
        for k, v in self.scheduler_config.items():
            name += f"_{k[0]}_{v:.4e}" if isinstance(v, float) else f"_{k[0]}_{v}"
        return name

    def gen_tags(self):
        return [
            self.net.split('.')[-1],
            *[f"{k[0]}={v}" for k, v in self.net_config.items()],
            self.optimizer.split('.')[-1],
            self.scheduler.split('.')[-1]
        ]

    def gen_config(self):
        return asdict(self)


def default_run_config():
    return RunConfig(
        project="JAX_Template",
        device="cuda:0",
        seeds=[42],
        net="model.MLP",
        optimizer="optax.adamw",
        scheduler="hyperbolic_lr.HyperbolicLR",
        epochs=50,
        batch_size=256,
        net_config={
            "nodes": 128,
            "layers": 4,
        },
        optimizer_config={
            "learning_rate": 1e-3,
        },
        scheduler_config={
            "upper_bound": 300,
            "max_iter": 50,
            "init_lr": 1e-3,
            "infimum_lr": 1e-5,
        },
    )


@dataclass
class OptimizeConfig:
    study_name: str
    trials: int
    seed: int
    metric: str
    direction: str
    sampler: dict = field(default_factory=dict)
    pruner: dict = field(default_factory=dict)
    search_space: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return cls(**config)

    def to_yaml(self, path):
        with open(path, 'w') as file:
            yaml.dump(asdict(self), file, sort_keys=False)

    def _create_sampler(self):
        module_name, class_name = self.sampler["name"].rsplit('.', 1)
        module = importlib.import_module(module_name)
        sampler_class = getattr(module, class_name)
        sampler_kwargs = self.sampler.get('kwargs', {})
        if class_name == "GridSampler":
            sampler_kwargs['search_space'] = self.grid_search_space()
        return sampler_class(**sampler_kwargs)

    def _create_pruner(self):
        if not self.pruner:
            return None
        module_name, class_name = self.pruner["name"].rsplit('.', 1)
        module = importlib.import_module(module_name)
        pruner_class = getattr(module, class_name)
        pruner_kwargs = self.pruner.get('kwargs', {})
        return pruner_class(**pruner_kwargs)

    def create_study(self, project):
        sampler = self._create_sampler()
        study = {
            'study_name': self.study_name,
            'storage': f'sqlite:///{project}.db',
            'sampler': sampler,
            'direction': self.direction,
            'load_if_exists': True,
        }
        pruner = self._create_pruner()
        if pruner:
            study['pruner'] = pruner
        return optuna.create_study(**study)

    def suggest_params(self, trial):
        params = {}
        for category, config in self.search_space.items():
            params[category] = {}
            for param, param_config in config.items():
                if param_config['type'] == 'int':
                    params[category][param] = trial.suggest_int(f"{category}_{param}", 
                                                                param_config['min'], 
                                                                param_config['max'], 
                                                                step=param_config.get('step', 1))
                elif param_config['type'] == 'float':
                    if param_config.get('log', False):
                        params[category][param] = trial.suggest_float(f"{category}_{param}", 
                                                                      param_config['min'], 
                                                                      param_config['max'],
                                                                      log=True)
                    else:
                        params[category][param] = trial.suggest_float(f"{category}_{param}", 
                                                                      param_config['min'], 
                                                                      param_config['max'])
                elif param_config['type'] == 'categorical':
                    params[category][param] = trial.suggest_categorical(f"{category}_{param}", 
                                                                        param_config['choices'])
        return params

    def grid_search_space(self):
        params = {}
        for category, config in self.search_space.items():
            for param, param_config in config.items():
                if param_config['type'] == 'categorical':
                    params[f"{category}_{param}"] = param_config['choices']
                else:
                    raise ValueError(f"Unsupported grid search space type: {param_config['type']}")
        return params


def abbreviate(s: str):
    return ''.join([w for w in s if w.isupper()])
