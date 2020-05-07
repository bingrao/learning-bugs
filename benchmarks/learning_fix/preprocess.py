from torch.utils.data import Dataset
from nmt.utils.context import Context, create_dir
from os.path import join
import collections
from torchtext import data

START_TOKEN = '<Start>'
END_TOKEN = '<End>'
UNK_TOKEN = 0


class Vocabulary:
    def __init__(self,
                 ctx,
                 dataset="small",
                 target="buggy",
                 min_frequency=0,
                 max_vocab_size=250,
                 downcase=False,
                 delimiter=" ",
                 embedding=None):
        assert target != "buggy" or target != "fixed"
        self.context = ctx
        self.logger = ctx.logger
        self.start_token = '<Start>'
        self.end_token = '<End>'
        self.raw_dir = join(self.context.project_raw_dir, dataset)
        self.processed_dir = join(self.context.project_processed_dir, dataset)
        create_dir(self.processed_dir)
        self.token_embedding = embedding
        self.target = target
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size
        self.downcase = downcase
        self.delimiter = delimiter
        self.data_with_count = self._build_vocab()  # [('key', 100), ..., ('for', 1)]
        self.data = [word for word, _ in self.data_with_count]  # ['key', ..., 'for', 1]
        self.unk_token = len(self.data)

    def __len__(self):
        return len(self.data) + 1

    def get_vocab_size(self):
        return self.__len__()

    def get_token_embedding(self, token):
        return self.token_embedding(token) \
            if self.token_embedding is not None \
            else self.data.index(token) if token in self.data else self.unk_token

    def add_start_end_token(self, seq_tokens):
        return [self.start_token] + seq_tokens + [self.end_token]

    def _build_vocab(self):
        self.logger.info(f"Building vocab {self.target} with max size {self.max_vocab_size}")

        # Counter for all tokens in the vocabulary
        cnt = collections.Counter()
        source_path = join(self.raw_dir, "train", f"{self.target}.txt")
        with open(source_path) as file:
            for line in file:
                if self.downcase:
                    line = line.lower()
                if self.delimiter == "":
                    tokens = list(line.strip())
                else:
                    tokens = line.strip().split(self.delimiter)
                tokens = [_ for _ in tokens if len(_) > 0]
                if self.target == "fixed":
                    tokens = [self.start_token] + tokens + [self.end_token]
                cnt.update(tokens)

        self.logger.info("Found %d unique tokens in the vocabulary.", len(cnt))

        # Filter tokens below the frequency threshold
        if self.min_frequency > 0:
            filtered_tokens = [(w, c) for w, c in cnt.most_common() if c > self.min_frequency]
            cnt = collections.Counter(dict(filtered_tokens))

        self.logger.info("Found %d unique tokens with frequency > %d.", len(cnt), self.min_frequency)

        # Sort tokens by 1. frequency 2. lexically to break ties
        word_with_counts = cnt.most_common()
        word_with_counts = sorted(word_with_counts, key=lambda x: (x[1], x[0]), reverse=True)

        # Take only max-vocab
        if self.max_vocab_size is not None:
            word_with_counts = word_with_counts[:self.max_vocab_size]

        if self.context.isDebug:
            with open(join(self.processed_dir, f"vocab.{self.target}.txt"), 'w') as file:
                for idx, (word, count) in enumerate(word_with_counts):
                    file.write(f'{idx}\t{word}\t{count}\n')
        return word_with_counts


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
        src, tgt = self.data[idx]
        return src, tgt

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
                # src_feature = Variable(torch.tensor(src_index).unsqueeze(0), requires_grad=False)

                tgt_tokens = self.tgt_vocab.add_start_end_token(list(tgt_line.split()))
                tgt_index = [self.tgt_vocab.get_token_embedding(token) for token in tgt_tokens]
                # tgt_feature = Variable(torch.tensor(tgt_index).unsqueeze(0), requires_grad=False)

                if self.context.isDebug:
                    f1.write(f'{src_line}\t{tgt_line}\n')
                    f2.write(f'{src_tokens}\t{tgt_tokens}\n')
                    f3.write(f'{src_index}\t{tgt_index}\n')

                self.data.append((src_index, tgt_index))


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
