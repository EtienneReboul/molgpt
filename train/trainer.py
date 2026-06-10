"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
import os
from contextlib import nullcontext

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from utils import check_novelty, sample, canonic_smiles, get_mol
import re
import pandas as pd
from rdkit import Chem


logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    # accumulate gradients over N micro-batches; effective_batch = batch_size * grad_accumulation_steps
    grad_accumulation_steps = 1
    # resume training from an existing checkpoint at ckpt_path
    resume = False

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, stoi, itos):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever device is available on the system (CUDA -> MPS -> CPU)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.stoi = stoi
        self.itos = itos

        self.model = self.model.to(self.device)

    def save_checkpoint(self, epoch, optimizer, best_loss):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_dir = os.path.dirname(self.config.ckpt_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        logger.info("saving %s", self.config.ckpt_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'tokens': self.tokens,
        }, self.config.ckpt_path)

    def load_checkpoint(self, optimizer):
        checkpoint = torch.load(self.config.ckpt_path, map_location=self.device, weights_only=True)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        if 'model_state_dict' in checkpoint:
            # new format: full training state
            raw_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            self.tokens = checkpoint.get('tokens', 0)
        else:
            # legacy format: checkpoint is the bare state dict (no epoch/optimizer state)
            logger.warning(
                "checkpoint is in legacy format (weights only) — optimizer state and epoch "
                "count are unavailable; resuming weights from epoch 0 with fresh optimizer"
            )
            raw_model.load_state_dict(checkpoint)
            start_epoch = 0
            best_loss = float('inf')

        logger.info("resuming from epoch %d", start_epoch)
        return start_epoch, best_loss

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = torch.amp.GradScaler('cuda', enabled=self.device.type == 'cuda')

        self.tokens = 0
        start_epoch = 0
        best_loss = float('inf')

        if config.resume and config.ckpt_path and os.path.exists(config.ckpt_path):
            start_epoch, best_loss = self.load_checkpoint(optimizer)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            lr = config.learning_rate

            if is_train:
                model.zero_grad()

            for it, (x, y, p, scaffold) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                p = p.to(self.device)
                scaffold = scaffold.to(self.device)

                # forward the model
                amp_context = torch.amp.autocast(device_type='cuda') if self.device.type == 'cuda' else nullcontext()
                with amp_context:
                    with torch.set_grad_enabled(is_train):
                        logits, loss, _ = model(x, y, p, scaffold)
                        loss = loss.mean()
                        losses.append(loss.item())

                if is_train:
                    # scale loss before backward so gradients are averaged over accumulation steps
                    scaled_loss = loss / config.grad_accumulation_steps
                    scaler.scale(scaled_loss).backward()

                    is_last_batch = (it + 1) == len(loader)
                    if ((it + 1) % config.grad_accumulation_steps == 0) or is_last_batch:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        model.zero_grad()

                        # decay the learning rate based on our progress
                        if config.lr_decay:
                            self.tokens += (y >= 0).sum() # number of tokens processed this step
                            if self.tokens < config.warmup_tokens:
                                # linear warmup
                                lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                            else:
                                # cosine learning rate decay
                                progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr = config.learning_rate * lr_mult
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr

                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if is_train:
                return float(np.mean(losses))

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        molecules = []

        for epoch in range(start_epoch, config.max_epochs):

            train_loss = run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                print(f'Saving at epoch {epoch + 1}')
                self.save_checkpoint(epoch, optimizer, best_loss)

            if self.config.generate:
                pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
                regex = re.compile(pattern)
                context = "C"
                for i in range(2):
                    x = torch.tensor([self.stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(512, 1).to(self.device)
                    p = None
                    sca = None
                    y = sample(model, x, self.config.block_size, temperature=0.8, sample=True, top_k=10, prop = p, scaffold = sca)
                    for gen_mol in y:
                        completion = ''.join([self.itos[int(i)] for i in gen_mol])
                        completion = completion.replace('<', '')
                        mol = get_mol(completion)
                        if mol:
                            smiles = Chem.MolToSmiles(mol)
                            molecules.append((mol, smiles, epoch))

        if self.config.generate:
            df = pd.DataFrame(molecules, columns = ['molecule', 'smiles', 'epoch'])
            return df

        return None
