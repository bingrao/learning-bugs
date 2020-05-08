from torch.utils.data import Dataset
from nmt.utils.context import Context, create_dir
from os.path import join
from torchtext import data
from benchmarks.learning_fix.Vocabulary import Vocabulary
import numpy as np

class LFDataset(Dataset):
    def __init__(self,
                 ctx=None,
                 target="train",
                 dataset="small",
                 embedding=None,
                 padding=0,
                 src_vocab=None,
                 tgt_vocab=None):
        assert target != "train" or target != "eval" or target != "test"
        self.context = ctx
        self.logger = ctx.logger
        self.target = target
        self.dataset = dataset
        self.raw_dir = join(self.context.project_raw_dir, self.dataset)
        self.processed_dir = join(self.context.project_processed_dir, self.dataset)
        create_dir(self.processed_dir)

        self.fields = data.Field(tokenize=lambda x: list(x.split()))
        self.padding = padding
        self.min_frequency = 0
        self.max_vocab_size = 500
        self.token_embedding = embedding

        self.src_vocab = src_vocab if src_vocab is not None \
            else Vocabulary(ctx=self.context,
                            dataset=self.dataset,
                            target="buggy",
                            min_frequency=self.min_frequency,
                            max_vocab_size=self.max_vocab_size,
                            embedding=self.token_embedding)

        self.tgt_vocab = tgt_vocab if tgt_vocab is not None \
            else Vocabulary(ctx=self.context,
                            dataset=self.dataset,
                            target="fixed",
                            min_frequency=self.min_frequency,
                            max_vocab_size=self.max_vocab_size,
                            embedding=self.token_embedding)
        self.data = []
        # load data
        self.load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a batch representation with
        # src, tgt_in, tgt_out and masks
        src, src_pos, tgt, tgt_pos = self.data[idx]
        return src, src_pos, tgt, tgt_pos

    def load(self):
        src_path = join(self.raw_dir, self.target, "buggy.txt")
        tgt_path = join(self.raw_dir, self.target, "fixed.txt")
        raw_data_path = join(self.processed_dir, f'{self.target}-raw.txt')
        raw_token_path = join(self.processed_dir, f'{self.target}-token.txt')
        raw_idx_path = join(self.processed_dir, f'{self.target}-index.txt')

        with open(src_path) as src_file:
            src_raw_data = src_file.readlines()
        with open(tgt_path) as tgt_file:
            tgt_raw_data = tgt_file.readlines()

        with open(raw_data_path, 'w') as f1, \
                open(raw_token_path, 'w') as f2, \
                open(raw_idx_path, 'w') as f3:

            for src_line, tgt_line in zip(src_raw_data, tgt_raw_data):
                src_tokens = list(src_line.split())
                src_index = [self.src_vocab.get_token_embedding(token) for token in src_tokens]
                src_pos = [idx for idx, _ in enumerate(src_index)]
                # src_feature = Variable(torch.tensor(src_index).unsqueeze(0), requires_grad=False)

                tgt_tokens = self.tgt_vocab.add_start_end_token(list(tgt_line.split()))
                tgt_index = [self.tgt_vocab.get_token_embedding(token) for token in tgt_tokens]
                tgt_pos = [idx for idx, _ in enumerate(tgt_index)]
                # tgt_feature = Variable(torch.tensor(tgt_index).unsqueeze(0), requires_grad=False)

                if self.context.isDebug:
                    f1.write(f'{src_line}\t{tgt_line}\n')
                    f2.write(f'{src_tokens}\t{tgt_tokens}\n')
                    f3.write(f'{src_index}\t{tgt_index}\n')

                self.data.append((src_index, src_pos, tgt_index, tgt_pos))


def dataset_generation(context, data_type="small"):
    logger = context.logger
    pad_idx = context.padding_index

    logger.info(f"Preparing train dataset ... ")
    train_dataset = LFDataset(ctx=context, target="train",
                              dataset=data_type, padding=pad_idx, src_vocab=None, tgt_vocab=None)

    src_vocab = train_dataset.src_vocab
    tgt_vocab = train_dataset.tgt_vocab

    logger.info(f"Preparing eval dataset ... ")
    eval_dataset = LFDataset(ctx=context, target="eval",
                             dataset=data_type, padding=pad_idx, src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    logger.info(f"Preparing test dataset ... ")
    test_dataset = LFDataset(ctx=context, target="test",
                             dataset=data_type, padding=pad_idx, src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    return train_dataset, eval_dataset, test_dataset


if __name__ == "__main__":
    dataset_generation(Context("Learning-fix based on Transformer"))
