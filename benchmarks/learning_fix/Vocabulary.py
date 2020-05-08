from nmt.utils.context import create_dir
from os.path import join
import collections


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
        self.word_with_count = self._build_vocab()  # [('key', 100), ..., ('for', 1)]
        self.words = [word for word, _ in self.word_with_count]  # ['key', ..., 'for', 1]
        self.word2idx = dict((w, i) for i, w in enumerate(self.words))
        self.idx2word = dict((i, w) for i, w in enumerate(self.words))

        self.unk_token = len(self.words)

    def __len__(self):
        return len(self.words) + 1

    def get_vocab_size(self):
        return self.__len__()

    def get_token_embedding(self, token):
        return self.token_embedding(token) \
            if self.token_embedding is not None \
            else self.words.index(token) if token in self.words else self.unk_token

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
