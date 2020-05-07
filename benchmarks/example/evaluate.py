from benchmarks.example.predict import Predictor
from nmt.model.transformer.model import build_model
from benchmarks.example.datasets import TranslationDataset, IndexDictionary
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from tqdm import tqdm
from nmt.utils.context import Context
import os

class Evaluator:
    def __init__(self, pred=None, save_filepath=None):

        self.predictor = pred
        self.save_filepath = save_filepath

    def evaluate_dataset(self, test_dataset):
        def tokenize(x):
            return x.split()

        predictions = []
        for source, target in tqdm(test_dataset):
            prediction = self.predictor.predict_one(source, num_candidates=1)[0]
            predictions.append(prediction)

        hypotheses = [tokenize(prediction) for prediction in predictions]
        list_of_references = [[tokenize(target)] for source, target in test_dataset]
        smoothing_function = SmoothingFunction()

        with open(self.save_filepath, 'w') as file:
            for (source, target), prediction, hypothesis, references in zip(test_dataset, predictions,
                                                                            hypotheses, list_of_references):
                sentence_bleu_score = sentence_bleu(references,
                                                    hypothesis,
                                                    smoothing_function=smoothing_function.method3)
                line = "{bleu_score}\t{source}\t{target}\t|\t{prediction}".format(
                    bleu_score=sentence_bleu_score,
                    source=source,
                    target=target,
                    prediction=prediction
                )
                file.write(line + '\n')

        return corpus_bleu(list_of_references, hypotheses, smoothing_function=smoothing_function.method3)


if __name__ == "__main__":

    context = Context("Evaluation")
    logger = context.logger

    logger.info('Constructing dictionaries...')
    source_dictionary = IndexDictionary.load(context.proj_processed_dir,
                                             mode='source',
                                             vocabulary_size=context.vocabulary_size)
    target_dictionary = IndexDictionary.load(context.proj_processed_dir,
                                             mode='target',
                                             vocabulary_size=context.vocabulary_size)
    

    logger.info('Building model...')
    model = build_model(context, source_dictionary.vocabulary_size, target_dictionary.vocabulary_size)

    predictor = Predictor(ctx=context,
                          m=model,
                          src_dictionary=source_dictionary,
                          tgt_dictionary=target_dictionary)

    timestamp = datetime.now()
    if context.save_result is None:
        eval_filepath = '{logs_dir}/eval-{cfg}-time={timestamp}.csv'.format(
            logs_dir=os.path.dirname(context.project_log),
            cfg=context.proj_name,
            timestamp=timestamp.strftime("%Y_%m_%d_%H_%M_%S"))
    else:
        eval_filepath = context.save_result

    evaluator = Evaluator(pred=predictor, save_filepath=eval_filepath)

    logger.info('Evaluating...')
    test_datasets = TranslationDataset(context, context.phase, limit=1000)
    bleu_score = evaluator.evaluate_dataset(test_datasets)
    logger.info('Evaluation time :', datetime.now() - timestamp)
    logger.info("BLEU score :", bleu_score)