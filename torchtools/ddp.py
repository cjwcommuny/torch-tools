from torch.nn.parallel import DistributedDataParallel

class DDP(DistributedDataParallel):
    def __getattr__(self, item):
        return getattr(self.module, item)

    def state_dict(self, destination=None, prefix='', keep_vars=False) -> dict:
        return self.module.state_dict(destination, prefix, keep_vars)

    