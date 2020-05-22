import torch
from torch import nn
import math
from torch.autograd import Variable


class PositionalEncodingDebug(torch.nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout_prob (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, num_embeddings, embedding_dim, dim, dropout_prob=0., padding_idx=0, max_len=5000):
        super(PositionalEncodingDebug, self).__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embbedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.weight = self.embbedding.weight
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        x = self.embbedding(x)
        x = x * math.sqrt(self.dim)
        if step is None:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x


# Position of input source/target word embedding
class PositionalEncoding(torch.nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       d_model (int): embedding size  d_model
    """

    def __init__(self, ctx, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.context = ctx
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # dim: (max_len, d_model)

        # In a sentense, it consists of several words which is indexed from 0.
        # Here max_len means the max number of words can hold by a input sentense.
        # We create refer table [[pe]] with 3D dimension (1, max_len, d_model),
        position = torch.arange(0, max_len).unsqueeze(1)  # dim: (max_len, 1)
        div_term = torch.exp((torch.arange(0, d_model, 2) *      # tensor([ 0,  2,  4, ..., d_model])
                             -(math.log(10000.0) / d_model)).float())  #
        # In index of numpy or tensor, start:end:step  0:d_model:2 = 0:-1:2
        # position.float() * div_term --> dim: (max_len, d_model/2)
        pe[:, 0::2] = torch.sin(position.float() * div_term)  # Replace values in the even position of cols: [0, 2, ..]
        pe[:, 1::2] = torch.cos(position.float() * div_term)  # Replace values in the odd positions of cols: [1, 3, ..]
        pe = pe.unsqueeze(0)  # dim: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x, step=None, position=None):
        """
        :param x: 3D deminsion input with a batch of input (batch_size, seq_len, d_model).
                  We can see the input like: there are [[batch_size]] size of sentence,
                  In each sentence, there are [[seq_len]] size of words
                  each word is embedded as 1D [[d_model]] dimension feature vector.
        :param step:
        :param position:
        :return:
        """

        if step is None:
            if position is not None and self.context.position_style == 'sequence':

                """
                We argue that tokens' positions usually are in a customized order, rather than 
                in a sequential order, for example in a tree-based order, which make positional 
                embedding more meaningful. In this case, we require programmers to provide position
                information for each sentence/code as a list of indexed number, for example [0, 3, 1, 5, 4, ...].
                As we can see here, the indexed numbers are not in a sequential order. Then we refer
                [[self.pe]] to look up the corresponding positional embedding and add them to the input x.
                """
                batch_size, seq_len = position.size()  # dim: (batch_size, seq_len)
                pos_embedding = torch.cat([self.pe[:, position[i, :]] for i in range(batch_size)])
                x = x + Variable(pos_embedding, requires_grad=False)

            elif position is not None and self.context.position_style == 'tree':
                """
                In each row of [[position]] is a list of list, that is [[node_pos1], [node_pos2], 
                [node_pos3], [node_pos4], ...]. In each cell of a [[position]] is a vector of poistion encoding for a 
                token. Since the size of vectors of each token may be different, so we need to pad them to 
                same dimension: d_model
                """

                x = x + Variable(position, requires_grad=False)

            elif position is not None and self.context.position_style == 'path':
                x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) + Variable(position, requires_grad=False)

            else:
                """
                By default, a position of tokens in a sentence/code is in a sequential order and 
                indexed by [0, 1, 2, ..., seq_len-1]. So in our positional embedding, we just
                take first "seq_len" positional embedding value in pe, and add them to 
                corresponding x embedding as a combination embedding representation of input x
                """
                x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        else:
            x = x + Variable(self.pe[:, step])
        return self.dropout(x)


# Source and target input embedding
class Embeddings(torch.nn.Module):
    def __init__(self, ctx, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.context = ctx
        self.lut = torch.nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        :param x: The input 2D dimension (batch_size, seq_len)
        :return: The output with 3D (batch_size, seq_len, d_model)
        """
        return self.lut(x) * math.sqrt(self.d_model)


class TransformerEmbeddings(torch.nn.Module):
    def __init__(self, ctx, d_model, vocab_size, dropout, max_len=1000):
        super(TransformerEmbeddings, self).__init__()
        self.context = ctx,
        self.d_model = d_model
        self.embedding = Embeddings(self.context, d_model, vocab_size)
        self.positional_embedding = PositionalEncoding(ctx, d_model, dropout, max_len)

    def forward(self, x, step=None, position=None):
        x_embedding = self.embedding(x)
        x_embedding_with_pos = self.positional_embedding(x_embedding, step=step, position=position)
        return x_embedding_with_pos
