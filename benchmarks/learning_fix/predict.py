from nmt.model.transformer.model import build_model
from utils.context import Context
from nmt.utils.pad import make_std_mask
from benchmarks.learning_fix.preprocess import dataset_generation
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Beam:

    def __init__(self, ctx=None, beam_size=4, min_length=0, n_top=1, ranker=None,
                 start_token_id=8, end_token_id=9):
        self.context = ctx
        self.beam_size = beam_size
        self.min_length = min_length
        self.ranker = ranker

        self.end_token_id = end_token_id
        self.top_sentence_ended = False

        self.prev_ks = []
        self.next_ys = [torch.LongTensor(beam_size).fill_(start_token_id)]  # remove padding

        self.current_scores = torch.FloatTensor(beam_size).zero_()
        self.all_scores = []

        # The attentions (matrix) for each time.
        self.all_attentions = []

        self.finished = []

        # Time and k pair for finished.
        self.finished = []
        self.n_top = n_top

        self.ranker = ranker

    def advance(self, next_log_probs, current_attention):
        # next_probs : (beam_size=8, d_model=128)
        # current_attention: (beam_size=8, #heads=8, seq_len=1, d_model=128)

        vocabulary_size = next_log_probs.size(1)
        # current_beam_size = next_log_probs.size(0)

        current_length = len(self.next_ys)
        if current_length < self.min_length:
            for beam_index in range(len(next_log_probs)):
                next_log_probs[beam_index][self.end_token_id] = -1e10

        if len(self.prev_ks) > 0:
            beam_scores = next_log_probs + self.current_scores.unsqueeze(1).expand_as(next_log_probs)
            # Don't let EOS have children.
            last_y = self.next_ys[-1]
            for beam_index in range(last_y.size(0)):
                if last_y[beam_index] == self.end_token_id:
                    beam_scores[beam_index] = -1e10 # -1e20 raises error when executing
        else:
            beam_scores = next_log_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        top_scores, top_score_ids = flat_beam_scores.topk(k=self.beam_size, dim=0, largest=True, sorted=True)

        self.current_scores = top_scores
        self.all_scores.append(self.current_scores)

        prev_k = top_score_ids / vocabulary_size  # (beam_size, )
        next_y = top_score_ids - prev_k * vocabulary_size  # (beam_size, )

        self.prev_ks.append(prev_k)
        self.next_ys.append(next_y)
        # for RNN, dim=1 and for transformer, dim=0.
        prev_attention = current_attention.index_select(dim=0, index=prev_k)  # (target_seq_len=1, beam_size, source_seq_len)
        self.all_attentions.append(prev_attention)


        for beam_index, last_token_id in enumerate(next_y):
            if last_token_id == self.end_token_id:
                # skip scoring
                self.finished.append((self.current_scores[beam_index], len(self.next_ys) - 1, beam_index))

        if next_y[0] == self.end_token_id:
            self.top_sentence_ended = True

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.next_ys[-1]

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    def done(self):
        return self.top_sentence_ended and len(self.finished) >= self.n_top

    def get_hypothesis(self, timestep, k):
        hypothesis, attentions = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hypothesis.append(self.next_ys[j + 1][k])
            # for RNN, [:, k, :], and for trnasformer, [k, :, :]
            attentions.append(self.all_attentions[j][k, :, :])
            k = self.prev_ks[j][k]
        attentions_tensor = torch.stack(attentions[::-1]).squeeze(1)  # (timestep, source_seq_len)
        return hypothesis[::-1], attentions_tensor

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                # global_scores = self.global_scorer.score(self, self.scores)
                # s = global_scores[i]
                s = self.current_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished = sorted(self.finished, key=lambda a: a[0], reverse=True)
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks


class Predictor:
    def __init__(self, ctx, m, src_dictionary, tgt_dictionary, max_length=50, beam_size=8):
        self.context = ctx
        self.logger = ctx.logger
        self.model = m
        self.source_dictionary = src_dictionary
        self.target_dictionary = tgt_dictionary
        self.start_token_id = self.target_dictionary.start_token_id
        self.end_token_id = self.target_dictionary.end_token_id
        self.max_length = max_length
        self.beam_size = beam_size
        self.attentions = None
        self.hypothesises = None

        # Set up evaluate model
        self.model.eval()
        # Loading model paramters from previous traning.
        if self.context.project_checkpoint is not None:
            self.checkpoint_filepath = self.context.project_checkpoint
            self.model.load_state_dict(torch.load(self.checkpoint_filepath, map_location='cpu'))
        else:
            self.context.logger.error("[%s] There is no module paramters input and please train it",
                                      self.__class__.__name__,)
            exit(-1)


    def predict(self, source=None, num_candidates=None):
        source = self.context.source if source is None else source
        num_candidates = self.context.num_candidates if num_candidates is None else num_candidates
        self.logger.debug("[%s] Predict Input Source: %s, nums of Candidnum_candidatesates %d",
                          self.__class__.__name__, str(source), num_candidates)

        # Get a list index number of a given input string based on source dictionary.
        # Input: a input string
        # output: a corresponding index list and the corresponding position

        source_embedding, source_position = self.source_dictionary.preprocess(source)

        self.logger.debug("[%s] The corresponding indexes of [%s]: %s, Position %s", self.__class__.__name__,
                          str(source), str(source_embedding), str(source_position))
        # dimision (1, seq_len)
        source_tensor = torch.tensor(source_embedding).unsqueeze(0)
        if source_position is not None:
            source_position = torch.tensor(source_position).unsqueeze(0)
        else:
            source_position = None
        length_tensor = torch.tensor(len(source)).unsqueeze(0) # (1, 1)
        self.logger.debug("[%s] The index source Tensor: %s, lenght %s", self.__class__.__name__,
                          source_tensor, length_tensor)

        # sources_mask = pad_masking(source_tensor, source_tensor.size(1))
        sources_mask = (source_tensor != 0).unsqueeze(-2)

        return self.predict_one(source_tensor, sources_mask, source_position, num_candidates)

    def predict_one(self, source, sources_mask, source_position, num_candidates):

        source = self.context.mapping_to_cuda(source)
        source_position = self.context.mapping_to_cuda(source_position)
        sources_mask = self.context.mapping_to_cuda(sources_mask)

        # dimision (1, seq_len, d_model)
        memory = self.model.encode(source, sources_mask, source_position)

        self.logger.debug("[%s] Encoder Source %s, Output %s dimensions", self.__class__.__name__,
                          source.size(), memory.size())

        # memory_mask = pad_masking(source, 1)
        memory_mask = sources_mask

        # Repeat beam_size times
        # (beam_size, seq_len, d_model)
        memory_beam = memory.detach().repeat(self.beam_size, 1, 1)

        self.logger.debug("[%s] Memory %s dimension", self.__class__.__name__, memory_beam.size())

        beam = Beam(ctx=self.context,
                    beam_size=self.beam_size,
                    min_length=0,
                    n_top=num_candidates,
                    ranker=None,
                    start_token_id=self.start_token_id,
                    end_token_id=self.end_token_id)

        for _ in range(self.max_length):
            # Generated a new token with dimension (beam_size, seq_len=1)
            new_inputs = beam.get_current_state().unsqueeze(1)
            new_mask = make_std_mask(new_inputs, 0)
            new_inputs = self.context.mapping_to_cuda(new_inputs)
            new_mask = self.context.mapping_to_cuda(new_mask)

            # dimension (bean_size, 1, d_model)
            decoder_outputs = self.model.decode(tgt=new_inputs,
                                                memory=memory_beam,
                                                memory_mask=memory_mask,
                                                tgt_mask=new_mask)

            self.logger.debug("[%s] Decoder Input %s, output %s dimensions",
                              self.__class__.__name__, new_inputs.size(), decoder_outputs.size())
            # dimension (beam_size, #heads, seq_len, d_model)
            attention = self.model.decoder.layers[-1].src_attn.attention
            self.logger.debug("[%s] attention %s dimension", self.__class__.__name__, attention.size())

            beam.advance(decoder_outputs.squeeze(1), attention)
            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=num_candidates)
        hypothesises, attentions = [], []
        for i, (times, k) in enumerate(ks[:num_candidates]):
            hypothesis, attention = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)
            attentions.append(attention)

        self.attentions = attentions
        self.hypothesises = [[token.item() for token in h] for h in hypothesises]
        hs = [self.target_dictionary.postprocess(h) for h in self.hypothesises]
        return list(reversed(hs))


if __name__ == "__main__":

    context = Context(desc="Prediction")
    logger = context.logger
    data_source_type = "small"
    logger.info(f"Loading {data_source_type} data from disk and parse it as a bunch of batches ...")
    train_dataset, eval_dataset, test_dataset = dataset_generation(context, data_type=data_source_type)

    logger.info('Constructing dictionaries...')
    source_dictionary = train_dataset.src_vocab
    target_dictionary = train_dataset.tgt_vocab

    logger.info('Building model...')
    model = build_model(context, len(source_dictionary), len(target_dictionary))

    logger.info("Building Predictor ....")
    predictor = Predictor(ctx=context,
                          m=model,
                          src_dictionary=source_dictionary,
                          tgt_dictionary=target_dictionary)

    logger.info("Get Predict Result ...")
    for index, candidate in enumerate(predictor.predict()):
        logger.info(f'Candidate {index} : {candidate}')
