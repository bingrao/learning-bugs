from torch.utils.data import Dataset
from torchtext import data
from benchmarks.learning_fix.Vocabulary import Vocabulary
from utils.context import Context, create_dir
from torch.utils.data import DataLoader
from nmt.data.batch import Batch
from os.path import join, exists
import torch
import numpy as np

PAD_INDEX = 0

class DataObject:
    def __init__(self, src, src_pos, tgt, tgt_pos, d_model):
        self.d_model = d_model
        self.src = src
        self.src_pos = src_pos
        if self.src_pos is not None:
            self.src_pos_dim = [self.expand(pos) for pos in self.src_pos]
        self.tgt = tgt
        self.tgt_pos = tgt_pos
        if self.tgt_pos is not None:
            self.tgt_pos_dim = [self.expand(pos) for pos in self.tgt_pos]

    def expand(self, position):
        if type(position) is list:
            index = list(filter(lambda x: x < self.d_model, position[1:]))
            embedding = np.zeros(self.d_model, dtype=int)
            embedding[index] = 1
            return embedding
        else:
            position

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
        # src, src_pos, tgt, tgt_pos = self.data[idx]
        return self.data[idx]


    def load(self):
        src_path = join(self.raw_dir, self.target, "buggy.txt")
        tgt_path = join(self.raw_dir, self.target, "fixed.txt")
        raw_embedding_path = join(self.processed_dir, f'{self.target}-embedding.txt')
        raw_pos_path = join(self.processed_dir, f'{self.target}-pos.txt')

        def _parse_data():
            self.logger.info(f"Parsing {self.target} data from raw data ...")
            raw_data_path = join(self.processed_dir, f'{self.target}-raw.txt')
            raw_token_path = join(self.processed_dir, f'{self.target}-token.txt')

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
                src_tokens_position = list(map(lambda x: x.split("@"), list(src_line.split())))
                src_tokens = [token[0] for token in src_tokens_position]
                src_embedding = [self.src_vocab.get_token_embedding(token) for token in src_tokens]

                # In a sequence, we can use their indexes as the postion,
                # In a tree architecture, we use node postion in the AST as correponding tokens' position
                if self.context.position_style == 'sequence':
                    src_pos = [idx for idx, _ in enumerate(src_embedding)]
                elif self.context.position_style == 'tree' or self.context.position_style == 'path':
                    # A position for a token is a list of nums, for example:
                    # void --> [57, 1, 5, 7, 45] means the position vector size is 57,
                    # and the index of [1, 5, 7, 45] are 1, the others are all zeros.
                    src_pos = [list(eval(token[-1])) for token in src_tokens_position]
                else:
                    src_pos = None


                tgt_tokens_position = list(map(lambda x: x.split("@"), list(tgt_line.split())))
                tgt_tokens = self.tgt_vocab.wrapping_tokens([token[0] for token in tgt_tokens_position])
                tgt_embedding = [self.tgt_vocab.get_token_embedding(token) for token in tgt_tokens]

                # In a sequence, we can use their indexes as the postion,
                # In a tree architecture, we use node postion in the AST as correponding tokens' position
                if self.context.position_style == 'sequence':
                    tgt_pos = [idx for idx, _ in enumerate(tgt_embedding)]
                elif self.context.position_style == 'tree' or self.context.position_style == 'path':
                    tgt_pos = [list(eval(token[-1])) for token in tgt_tokens_position]
                else:
                    tgt_pos = None

                # save intermedate data into files for Debug Purpose
                if src_line is not None:
                    raw_data.append((src_line, tgt_line))
                if src_tokens is not None:
                    raw_token.append((src_tokens, tgt_tokens))
                if src_embedding is not None:
                    raw_embedding.append((src_embedding, tgt_embedding))
                if src_pos is not None:
                    raw_position.append((src_pos, tgt_pos))

                self.data.append(DataObject(src_embedding, src_pos, tgt_embedding, tgt_pos, self.context.d_model))

            with open(raw_data_path, 'w') as f1, \
                    open(raw_token_path, 'w') as f2, \
                    open(raw_embedding_path, 'w') as f3, \
                    open(raw_pos_path, 'w') as f4:
                f1.write("\n".join(map(lambda x: f'{x[0]}\t{x[1]}', raw_data)))
                f2.write("\n".join(map(lambda x: f'{x[0]}\t{x[1]}', raw_token)))
                f3.write("\n".join(map(lambda x: f'{x[0]}\t{x[1]}', raw_embedding)))
                f4.write("\n".join(map(lambda x: f'{x[0]}\t{x[1]}', raw_position)))

        def _load():
            self.logger.info(f"Loading {self.target}  data from existing previous results ...")
            with open(raw_embedding_path) as src_file:
                embedding_data = src_file.readlines()

            if self.context.position_style == 'default':
                for embedding in embedding_data:
                    src_embedding = list(eval(embedding.split("\t")[0]))
                    tgt_embedding = list(eval(embedding.split("\t")[-1]))
                    src_pos = None
                    tgt_pos = None
                    self.data.append(DataObject(src_embedding, src_pos, tgt_embedding, tgt_pos, self.context.d_model))
            else:
                with open(raw_pos_path) as tgt_file:
                    positions_data = tgt_file.readlines()

                for embedding, position in zip(embedding_data, positions_data):
                    src_embedding = list(eval(embedding.split("\t")[0]))
                    tgt_embedding = list(eval(embedding.split("\t")[-1]))
                    src_pos = list(eval(position.split("\t")[0]))
                    tgt_pos = list(eval(position.split("\t")[-1]))
                    self.data.append(DataObject(src_embedding, src_pos, tgt_embedding, tgt_pos, self.context.d_model))





        if self.context.position_style == "default":
            if exists(raw_embedding_path):
                _load()
            else:
                _parse_data()
        else:
            if exists(raw_embedding_path) and exists(raw_pos_path):
                _load()
            else:
                _parse_data()

def collate_fn_default(batch):
    """
    merges a list of samples to form a mini-batch.
    batch: (src, src_pos, tgt, tgt_pos)
    By default, there is no positions for src and tgt
    """

    src_max_len = max([len(ele.src) for ele in batch])
    tgt_max_len = max([len(ele.tgt) for ele in batch])

    srcs_padded = [ele.src + [PAD_INDEX] * (src_max_len - len(ele.src)) for ele in batch]
    tgts_padded = [ele.tgt + [PAD_INDEX] * (tgt_max_len - len(ele.tgt)) for ele in batch]

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

    src_max_len = max([len(ele.src) for ele in batch])
    tgt_max_len = max([len(ele.tgt) for ele in batch])

    srcs_padded = [ele.src + [PAD_INDEX] * (src_max_len - len(ele.src)) for ele in batch]
    tgts_padded = [ele.tgt + [PAD_INDEX] * (tgt_max_len - len(ele.tgt)) for ele in batch]

    srcs_pos_padded = [ele.src_pos + list(range(max(src_pos) + 1, max(ele.src_pos) + 1 + (src_max_len - len(ele.src_pos))))
                       for ele in batch]
    tgts_pos_padded = [ele.tgt_pos + list(range(max(ele.tgt_pos) + 1, max(ele.tgt_pos) + 1 + (tgt_max_len - len(ele.tgt_pos))))
                       for ele in batch]

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

    src_max_len = max([len(ele.src) for ele in batch])
    tgt_max_len = max([len(ele.tgt) for ele in batch])

    srcs_padded = [ele.src + [PAD_INDEX] * (src_max_len - len(ele.src)) for ele in batch]
    tgts_padded = [ele.tgt + [PAD_INDEX] * (tgt_max_len - len(ele.tgt)) for ele in batch]



    srcs_pos_padded = [ele.src_pos_dim + [[0]*ele.d_model] * (src_max_len - len(ele.src_pos)) for ele in batch]
    tgts_pos_padded = [ele.tgt_pos_dim + [[0]*ele.d_model] * (tgt_max_len - len(ele.tgt_pos)) for ele in batch]

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
