import time

import torch

try:
    from lightning.pytorch import Callback
    from lightning.pytorch.utilities import rank_zero_info
except ImportError:
    from pytorch_lightning import Callback
    from pytorch_lightning.utilities import rank_zero_info


class CUDACallback(Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(self.root_gpu(trainer))
        torch.cuda.synchronize(self.root_gpu(trainer))
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(self.root_gpu(trainer))
        max_memory = torch.cuda.max_memory_allocated(self.root_gpu(trainer)) / 2 ** 20
        epoch_time = time.time() - self.start_time

        max_memory = trainer.training_type_plugin.reduce(max_memory)
        epoch_time = trainer.training_type_plugin.reduce(epoch_time)

        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")

    def root_gpu(self, trainer):
        return trainer.strategy.root_device.index
