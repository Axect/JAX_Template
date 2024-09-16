import jax
import jax.numpy as jnp


class HyperbolicLR:
    def __init__(self, upper_bound, max_iter, init_lr, infimum_lr):
        if upper_bound < max_iter:
            raise ValueError("upper_bound must be greater than max_iter")
        elif infimum_lr >= init_lr:
            raise ValueError("infimum_lr must be less than init_lr")

        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.init_lr = init_lr
        self.infimum_lr = infimum_lr

    def gen_scheduler(self):
        @jax.jit
        def scheduler(step):
            x = step
            N = self.max_iter
            U = self.upper_bound
            delta_lr = self.init_lr - self.infimum_lr
            return self.init_lr + delta_lr * (
                jnp.sqrt((N - x) / U * (2 - (N + x) / U)) - jnp.sqrt(N / U * (2 - N / U))
            )
        return scheduler


class ExpHyperbolicLR:
    def __init__(self, upper_bound, max_iter, init_lr, infimum_lr):
        if upper_bound < max_iter:
            raise ValueError("upper_bound must be greater than max_iter")
        elif infimum_lr >= init_lr:
            raise ValueError("infimum_lr must be less than init_lr")

        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.init_lr = init_lr
        self.infimum_lr = infimum_lr

    def gen_scheduler(self):
        @jax.jit
        def scheduler(step):
            x = step
            N = self.max_iter
            U = self.upper_bound
            lr_ratio = self.init_lr / self.infimum_lr
            return self.init_lr * lr_ratio ** (
                jnp.sqrt((N - x) / U * (2 - (N + x) / U)) - jnp.sqrt(N / U * (2 - N / U))
            )
        return scheduler
