from torch.nn.parallel import DistributedDataParallel

class DDP(DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def state_dict(self, destination=None, prefix='', keep_vars=False) -> dict:
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict)