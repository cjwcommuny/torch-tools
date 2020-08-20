from typing import Callable

from torch import nn
from torch.utils.tensorboard import SummaryWriter


def log_parameter_distributions(
        tensorboard: SummaryWriter,
        model: nn.Module,
        global_step: int,
        bins: str='tensorflow'
):
    for name, parameter in model.named_parameters():
        tensorboard.add_histogram(
            f"param/{name}",
            parameter,
            global_step,
            bins
        )


def need_forward_hook(cls):
    cls.__need_forward_hook__ = True
    return cls


def need_backward_hook(cls):
    """
    usage:
    @need_backward_hook
    class MySubModule(nn.Module):
        ...

    class MyModule(nn.Module):
        def __init__(self):
            self.layer = MySubModule()
        def forward(self, x):
            ...

    model = MyModule()

    :param cls:
    :return:
    """
    cls.__need_backward_hook__ = True
    return cls




def register_forward_hooks(module: nn.Module, HookType: Callable):
    for name, submodule in module.named_modules():
        if not hasattr(submodule, '__need_forward_hook__'):
            continue
        hook = HookType(name=name)
        submodule.register_forward_hook(hook)


def register_backward_hooks(module: nn.Module, HookType: Callable):
    for name, submodule in module.named_modules():
        if not hasattr(submodule, '__need_backward_hook__'):
            continue
        hook = HookType(name=name)
        submodule.register_backward_hook(hook)


class HookCounter:
    def __init__(self, init: int=-1):
        self.counter = init

    def __call__(self) -> int:
        return self.counter

    def increment(self):
        self.counter += 1



class ModuleHook:
    def __init__(self, name: str, tensorboard: SummaryWriter, counter: HookCounter, trigger_cycle: int):
        self.name = name
        self.tensorboard = tensorboard
        self.trigger_cycle = trigger_cycle
        self.counter = counter

    def log(self, data, tag: str):
        count = self.counter()
        if self.trigger_cycle is not None and count % self.trigger_cycle != 0:
            return
        assert self.name is not None
        if isinstance(data, tuple):
            for idx, tensor in enumerate(data):
                self.tensorboard.add_histogram(f'{tag}/{self.name}.data-{idx}', tensor, count)
        else:
            self.tensorboard.add_histogram(self.name, data, count)


class BackwardInputHook(ModuleHook):
    def __call__(self, module: nn.Module, grad_input, grad_output):
        return self.log(grad_input, tag='backward_in')

class ForwardOutputHook(ModuleHook):
    def __call__(self, module: nn.Module, input, output):
        return self.log(output, tag='forward_out')
