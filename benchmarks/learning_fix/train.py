import torch.nn as nn
from torch.autograd import Variable
import torch
from nmt.model.transformer.model import build_model
from utils.context import Context, create_dir
from benchmarks.learning_fix.preprocess import dataset_generation, generated_iter_dataset
import time
from datetime import datetime
import os
from os.path import join

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model.src_embed.d_model
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model, factor=1, warmup=2000,
                   optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class SimpleLossComputeWithLablSmoothing:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, devices=None, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm=1):
        # print(f" Before input x {x}, y {y}")
        x = self.generator(x)
        # print(f" After input x {x.size()}, y {y.size()}")
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        # print(f" Ouput x {x.size()}, loss {loss}")
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # https://github.com/pytorch/pytorch/issues/15585
        # return loss.data[0] * norm
        return loss.data.item() * norm


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.base_loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)

    def __call__(self, x, y, norm=1):
        batch_size, seq_len, vocabulary_size = x.size()

        outputs_flat = x.view(batch_size * seq_len, vocabulary_size)
        targets_flat = y.view(batch_size * seq_len)

        loss = self.base_loss_function(outputs_flat, targets_flat)
        count = (y != 0).sum().item()

        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss, count


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        # print(f" input x {x}, target {target.size()} self size {self.size}")
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # print(f"true_dist {true_dist}")
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class DataProcessEngine:
    def __init__(self, context):
        self.context = context
        self.logger = context.logger

        self.train_iter = None
        self.eval_iter = None
        self.test_iter = None

        self.src_vocab = None
        self.tgt_vocab = None

        self.model = None
        self.nums_batch = context.batch_size
        self.epochs = context.epochs
        self.padding_index = context.padding_index
        self.device = context.device  # cpu or gpu
        self.device_id = context.device_id  # [0, 1, 2, 3]

        self.save_every = context.save_every
        self.save_format = 'epoch={epoch:0>3}-val_loss={val_loss:<.3}.pth'
        self.epoch = 0
        self.history = []
        run_name = (
            "d_model={d_model}-"
            "layers_count={layers_count}-"
            "heads_count={heads_count}-"
            "pe={positional_encoding}-"
            "optimizer={optimizer}-"
            "{timestamp}"
        ).format(**ctx.config, timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.checkpoint = context.project_checkpoint
        self.checkpoint_dir = join(os.path.dirname(context.project_raw_dir), 'checkpoints', run_name)
        create_dir(self.checkpoint_dir)

    def preprocess(self, data_source_type="small"):

        self.logger.info(f"Loading {data_source_type} data from disk and parse it as a bunch of batches ...")
        train_dataset, eval_dataset, test_dataset = dataset_generation(self.context, data_type=data_source_type)

        self.logger.info("Build iteral dataset ... ")
        self.train_iter = generated_iter_dataset(self.context, train_dataset, self.nums_batch)
        self.eval_iter = generated_iter_dataset(self.context, eval_dataset, self.nums_batch)
        self.test_iter = generated_iter_dataset(self.context, test_dataset, self.nums_batch)

        self.logger.info("Build src/tgt Vocabulary ...")
        self.src_vocab = train_dataset.src_vocab
        self.tgt_vocab = train_dataset.tgt_vocab

        self.logger.info("Build transformer model ...")
        self.model = build_model(self.context, len(self.src_vocab), len(self.tgt_vocab))
        # self.model.cuda() if self.context.is_cuda else None
        self.logger.debug(self.model)

    def run(self, loss_func=None, opt=None):
        criterion = LabelSmoothing(size=len(self.tgt_vocab), padding_idx=self.padding_index, smoothing=0.1)
        criterion.cuda() if self.context.is_cuda else None

        self.logger.info("Training Process is begining ...")
        for epoch in range(self.epochs):
            # Set model in train
            self.epoch = epoch
            self.model.train()
            total_loss = self.run_epoch(self.train_iter, loss_func(self.model.generator, criterion, opt=opt))

            # Evaluation Model
            self.model.eval()
            # Get loss
            loss = self.run_epoch(self.eval_iter,
                                  loss_func(self.model.generator, criterion, opt=None))

            self.logger.info("Epoch Step: %d, \tLoss: %f", epoch, loss)

            if epoch % self.save_every == 0 or epoch == self.epochs:
                self._save_model(epoch, total_loss, loss)

    def run_epoch(self, data_iter, loss_compute):
        """
        Standard Training and Logging Function
        """
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(data_iter):

            src = self.context.mapping_to_cuda(batch.src)
            src_pos = self.context.mapping_to_cuda(batch.src_pos)
            trg = self.context.mapping_to_cuda(batch.trg)
            trg_pos = self.context.mapping_to_cuda(batch.trg_pos)
            trg_y = self.context.mapping_to_cuda(batch.trg_y)
            src_mask = self.context.mapping_to_cuda(batch.src_mask)
            trg_mask = self.context.mapping_to_cuda(batch.trg_mask)

            # Model forward and output result
            out = self.model(src=src,
                             tgt=trg,
                             src_mask=src_mask,
                             tgt_mask=trg_mask,
                             src_pos=src_pos,
                             tgt_pos=trg_pos)

            # Get loss for this iteration and backward weight to model
            loss = loss_compute(out, trg_y, batch.ntokens)

            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % 50 == 1:
                elapsed = time.time() - start
                self.logger.info("Iteration Step: %d, \tLoss: %f, \tTokens per Sec: %f",
                                 i, loss / batch.ntokens, tokens / elapsed)
                start = time.time()
                tokens = 0

        return total_loss / total_tokens


    def _save_model(self, epoch, train_epoch_loss, val_epoch_loss):

        checkpoint_filename = self.save_format.format(
            epoch=epoch,
            val_loss=val_epoch_loss
        )

        if self.checkpoint is None:
            checkpoint_filepath = join(self.checkpoint_dir, checkpoint_filename)
        else:
            checkpoint_filepath = self.checkpoint

        save_state = {
            'epoch': epoch,
            'train_loss': train_epoch_loss,
            'val_loss': val_epoch_loss,
            'checkpoint': checkpoint_filepath,
        }

        if self.epoch > 0:
            self.logger.info("Saved model to {}".format(checkpoint_filepath))
            torch.save(self.model.state_dict(), checkpoint_filepath)
            self.history.append(save_state)


    def _elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed).split('.')[0]  # remove milliseconds

    def postprocess(self):
        pass


if __name__ == "__main__":
    ctx = Context(desc="Learning-fix based on Transformer")
    logger = ctx.logger

    logger.info("Build Data Process Engine based on input parsed dataset ...")
    engine = DataProcessEngine(ctx)

    logger.info("Preparing dataset and build model for trani ...")
    engine.preprocess(data_source_type="small")

    logger.info("Training and evaluating the model ...")
    engine.run(loss_func=SimpleLossComputeWithLablSmoothing, opt=get_std_opt(engine.model))

    logger.info("Testing and data clean ...")
    engine.postprocess()
