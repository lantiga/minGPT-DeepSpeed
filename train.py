from argparse import ArgumentParser
import math
import numpy as np
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import lightning.pytorch as pl
    import lightning.pytorch as pl
    from lightning import Trainer
    from lightning import seed_everything
    from lightning.pytorch.strategies import DeepSpeedStrategy
    from lightning.pytorch.utilities import rank_zero_info
    from lightning.pytorch.utilities.meta import init_meta_context
except:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning import seed_everything
    from pytorch_lightning.strategies import DeepSpeedStrategy
    from pytorch_lightning.utilities import rank_zero_info
    from pytorch_lightning.utilities.meta import init_meta_context

try:
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
except:
    pass

from mingpt.callback import CUDACallback
from mingpt.lr_decay import LearningRateDecayCallback
from mingpt.block import Block


class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        rank_zero_info('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class GPT(pl.LightningModule):
    def __init__(self,
                 vocab_size,
                 weight_decay=0.1,
                 betas=(0.9, 0.95),
                 learning_rate=3e-4,
                 n_embd=768,
                 block_size=128,
                 embd_pdrop=0.1,
                 n_layer=12,
                 n_head=4,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1
                 ):
        super().__init__()
        self.save_hyperparameters()
        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # decoder head
        self.ln_f = nn.LayerNorm(self.hparams.n_embd)
        self.head = nn.Linear(self.hparams.n_embd, self.hparams.vocab_size, bias=False)

        self.block_size = self.hparams.block_size

        self.blocks = nn.ModuleList([Block(self.hparams) for _ in range(self.hparams.n_layer)])

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        if self.deepspeed_offload:
            # todo: need to enable deepspeed cpu adam only if offloading
            return DeepSpeedCPUAdam(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)

        if self.deepspeed:
            return FusedAdam(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)

        return torch.optim.Adam(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)

    @property
    def deepspeed(self) -> bool:
        strategy = self.trainer.strategy
        return isinstance(strategy, DeepSpeedStrategy)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config['zero_optimization']
            return config.get('offload_optimizer') or config.get('offload_param')
        return False

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        if self.deepspeed:
            for block in self.blocks:
                x = deepspeed.checkpointing.checkpoint(block, x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log('train_loss', loss)
        return loss


if __name__ == '__main__':
    seed_everything(42)

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--n_layer', default=22, type=int)
    parser.add_argument('--n_head', default=16, type=int)
    parser.add_argument('--n_embd', default=3072, type=int)
    parser.add_argument('--learning_rate', default=6e-4, type=float)
    parser.add_argument('--block_size', default=128, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--compile', default=0, type=int)
    args = parser.parse_args()

    if not os.path.exists("input.txt"):
        os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

    text = open('input.txt', 'r').read()  # don't worry we won't run out of file handles
    train_dataset = CharDataset(text, args.block_size)  # one line of poem is roughly 50 characters
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    with init_meta_context():
        model = GPT(
            vocab_size=train_dataset.vocab_size,
            block_size=train_dataset.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            learning_rate=args.learning_rate
        )

    if args.compile:
        model = torch.compile(model)

    lr_decay = LearningRateDecayCallback(
        learning_rate=6e-4,
        warmup_tokens=512 * 20,
        final_tokens=2 * len(train_dataset) * args.block_size
    )

    callbacks = [lr_decay]
    if torch.cuda.is_available():
        callbacks.append(CUDACallback())

    trainer = Trainer.from_argparse_args(
        args,
        max_epochs=10,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        precision=16,
    )
    trainer.fit(model, train_loader)
