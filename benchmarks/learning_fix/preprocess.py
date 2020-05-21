from torch.utils.data import Dataset
from torchtext import data
from benchmarks.learning_fix.Vocabulary import Vocabulary
from utils.context import Context, create_dir
from torch.utils.data import DataLoader
from nmt.data.batch import Batch
from os.path import join
import torch
import numpy as np

PAD_INDEX = 0
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

        # Debug
        raw_data = []
        raw_token = []
        raw_embedding = []
        raw_position = []

        with open(src_path) as src_file:
            src_raw_data = src_file.readlines()
        with open(tgt_path) as tgt_file:
            tgt_raw_data = tgt_file.readlines()

        for src_line, tgt_line in zip(src_raw_data, tgt_raw_data):
            src_tokens = list(src_line.split())
            src_embedding = [self.src_vocab.get_token_embedding(token) for token in src_tokens]

            # In a sequence, we can use their indexes as the postion,
            # In a tree architecture, we use node postion in the AST as correponding tokens' position
            src_pos = [idx for idx, _ in enumerate(src_embedding)]

            tgt_tokens = self.tgt_vocab.wrapping_tokens(list(tgt_line.split()))
            tgt_embedding = [self.tgt_vocab.get_token_embedding(token) for token in tgt_tokens]

            # In a sequence, we can use their indexes as the postion,
            # In a tree architecture, we use node postion in the AST as correponding tokens' position
            tgt_pos = [idx for idx, _ in enumerate(tgt_embedding)]

            # save intermedate data into files for Debug Purpose
            if self.context.isDebug:
                raw_data.append((src_line, tgt_line))
                raw_token.append((src_tokens, tgt_tokens))
                raw_embedding.append((src_embedding, tgt_embedding))
                raw_position.append((src_pos, tgt_pos))

            self.data.append((src_embedding, src_pos, tgt_embedding, tgt_pos))

        if self.context.isDebug:
            raw_data_path = join(self.processed_dir, f'{self.target}-raw.txt')
            raw_token_path = join(self.processed_dir, f'{self.target}-token.txt')
            raw_embedding_path = join(self.processed_dir, f'{self.target}-embedding.txt')
            raw_pos_path = join(self.processed_dir, f'{self.target}-pos.txt')

            with open(raw_data_path, 'w') as f1, \
                    open(raw_token_path, 'w') as f2, \
                    open(raw_embedding_path, 'w') as f3, \
                    open(raw_pos_path, 'w') as f4:
                f1.write("\n".join(map(lambda x: f'{x[0]}\t{x[1]}', raw_data)))
                f2.write("\n".join(map(lambda x: f'{x[0]}\t{x[1]}', raw_token)))
                f3.write("\n".join(map(lambda x: f'{x[0]}\t{x[1]}', raw_embedding)))
                f4.write("\n".join(map(lambda x: f'{x[0]}\t{x[1]}', raw_position)))

def collate_fn_default(batch):
    """
    merges a list of samples to form a mini-batch.
    batch: (src, src_pos, tgt, tgt_pos)
    By default, there is no positions for src and tgt
    """

    src_max_len = max([len(src) for src, _, _, _ in batch])
    tgt_max_len = max([len(tgt) for _, _, tgt, _ in batch])

    srcs_padded = [src + [PAD_INDEX] * (src_max_len - len(src)) for src, _, _, _ in batch]
    tgts_padded = [tgt + [PAD_INDEX] * (tgt_max_len - len(tgt)) for _, _, tgt, _ in batch]

    srcs_tensor = torch.tensor(srcs_padded)
    tgts_tensor = torch.tensor(tgts_padded)

    return Batch(src=srcs_tensor, trg=tgts_tensor, pad=PAD_INDEX)

def collate_fn_sequence(batch):
    """
    merges a list of samples to form a mini-batch.
    batch: (src, src_pos, tgt, tgt_pos)
    Here, src_pos and tgt_pos are a sequence of indexed number
    for each corresponding token.
    for example
        src_pos = [0, 1, 2, 3, 4, 5, 6, 8]
        tgt_pos = [0, 1, 2, 3, 4, 5, 6, 8]
    """

    src_max_len = max([len(src) for src, _, _, _ in batch])
    tgt_max_len = max([len(tgt) for _, _, tgt, _ in batch])

    srcs_padded = [src + [PAD_INDEX] * (src_max_len - len(src)) for src, _, _, _ in batch]
    tgts_padded = [tgt + [PAD_INDEX] * (tgt_max_len - len(tgt)) for _, _, tgt, _ in batch]

    srcs_pos_padded = [src_pos + list(range(max(src_pos) + 1, max(src_pos) + 1 + (src_max_len - len(src_pos))))
                       for _, src_pos, _, _ in batch]
    tgts_pos_padded = [tgt_pos + list(range(max(tgt_pos) + 1, max(tgt_pos) + 1 + (tgt_max_len - len(tgt_pos))))
                       for _, _, _, tgt_pos in batch]

    srcs_tensor = torch.tensor(srcs_padded)
    tgts_tensor = torch.tensor(tgts_padded)
    srcs_pos_tensor = torch.tensor(srcs_pos_padded)
    tgts_pos_tensor = torch.tensor(tgts_pos_padded)

    return Batch(src=srcs_tensor, trg=tgts_tensor, pad=PAD_INDEX, src_pos=srcs_pos_tensor, trg_pos=tgts_pos_tensor)

def collate_fn_tree(batch):
    """
    merges a list of samples to form a mini-batch.
    batch: (src, src_pos, tgt, tgt_pos)
    where src_pos/tgt_pos is a list of vectors positional embeddings
    In a list, a vector is a fixed-length vector for a corresponding token.
    However, the lenght of a vector may be different for different input source
    for example src_pos = [[1, 0, 2], [1, 2], ...]
    """

    def padding_vector(x):
        length = max(map(len, x))
        return np.array([xi + [0.0] * (length - len(xi)) for xi in x])

    src_max_len = max([len(src) for src, _, _, _ in batch])
    tgt_max_len = max([len(tgt) for _, _, tgt, _ in batch])

    srcs_padded = [src + [PAD_INDEX] * (src_max_len - len(src)) for src, _, _, _ in batch]
    tgts_padded = [tgt + [PAD_INDEX] * (tgt_max_len - len(tgt)) for _, _, tgt, _ in batch]

    default_dim_pos = 64
    empty_pos = [0.0] * default_dim_pos

    srcs_pos_alian = [src_pos + [empty_pos] * (src_max_len - len(src_pos)) for _, src_pos, _, _ in batch]
    tgts_pos_alian = [tgt_pos + [empty_pos] * (tgt_max_len - len(tgt_pos)) for _, _, _, tgt_pos in batch]

    srcs_pos_padded = [padding_vector(vector) for vector in srcs_pos_alian]
    tgts_pos_padded = [padding_vector(vector) for vector in tgts_pos_alian]

    srcs_tensor = torch.tensor(srcs_padded)
    tgts_tensor = torch.tensor(tgts_padded)

    srcs_pos_tensor = torch.tensor(srcs_pos_padded)
    tgts_pos_tensor = torch.tensor(tgts_pos_padded)

    return Batch(src=srcs_tensor, trg=tgts_tensor, pad=PAD_INDEX, src_pos=srcs_pos_tensor, trg_pos=tgts_pos_tensor)

def dataset_generation(context, data_type="small"):
    logger = context.logger
    pad_idx = context.padding_index

    logger.info(f"Preparing train {data_type} dataset ... ")
    train_dataset = LFDataset(ctx=context, target="train",
                              dataset=data_type, padding=pad_idx, src_vocab=None, tgt_vocab=None)

    src_vocab = train_dataset.src_vocab
    tgt_vocab = train_dataset.tgt_vocab

    logger.info(f"Preparing eval {data_type} dataset ... ")
    eval_dataset = LFDataset(ctx=context, target="eval",
                             dataset=data_type, padding=pad_idx, src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    logger.info(f"Preparing test {data_type} dataset ... ")
    test_dataset = LFDataset(ctx=context, target="test",
                             dataset=data_type, padding=pad_idx, src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    return train_dataset, eval_dataset, test_dataset

def generated_iter_dataset(context, dataset, nums_batch=64):

    if context.position_style == 'sequence':
        collate_fn = collate_fn_sequence
    elif context.position_style == 'tree':
        collate_fn = collate_fn_tree
    elif context.position_style == 'path':
        collate_fn = collate_fn_tree
    else:
        collate_fn = collate_fn_default

    return DataLoader(dataset,
                      batch_size=nums_batch,
                      shuffle=True,
                      collate_fn=collate_fn)


if __name__ == "__main__":
    dataset_generation(Context("Learning-fix based on Transformer"))
