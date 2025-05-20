# Gradual Warmup Scheduler
# Implemented from Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour https://arxiv.org/pdf/1706.02677.pdf
# Gradual increase in learning rate by a constant amount to avoid sudden increase in lr

import warnings

from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class GradualWarmupScheduler(_LRScheduler):
    """
    Sets the learning rate of parameter group to gradually increase for num_epochs from start_lr to the original lr set
    for the optimizer.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        eps_lr (float): min or starting learning rate which is gradually/linearly increased
            to optimizer lr. Default: 0.000001
        warmup_epochs (int): num of epochs during which the lr is increased. Default: 5.
        after_scheduler (Scheduler): scheduler to use after gradual warmup of lr is done. Default: None.
        last_epoch (int): The index of last epoch. Default: -1.
            verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.
    """

    def __init__(self, optimizer, eps_lr=0.000001, warmup_epochs=5, after_scheduler=None, last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.eps_lr = eps_lr
        self.after_scheduler = after_scheduler

        get_last_lr = getattr(self.after_scheduler, "get_last_lr", None)
        if not callable(get_last_lr):

            def get_last_lr():
                return [group["lr"] for group in self.optimizer.param_groups]

        self.after_scheduler.get_last_lr = get_last_lr

        self.finished = False  # set to True when warmup done
        super(GradualWarmupScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch > self.warmup_epochs:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr for base_lr in self.base_lrs]

        return [max(base_lr * (self.last_epoch / self.warmup_epochs), self.eps_lr) for base_lr in self.base_lrs]

    def step(self, metrics=None, epoch=None):
        # metrics is discarded unless ReduceLROnPlateau is used as after_scheduler
        # adopted from official pytorch _LRScheduler implementation
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the learning rate schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )
        self._step_count += 1
        if self.finished and self.after_scheduler:
            # if ReduceLROnPlateau is used, use the metrics parameter
            if isinstance(self.after_scheduler, ReduceLROnPlateau):
                return self.after_scheduler.step(metrics=metrics, epoch=epoch)
            if epoch is None:
                self.after_scheduler.step()
            else:
                self.after_scheduler.step(epoch=epoch - self.warmup_epochs)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step()
