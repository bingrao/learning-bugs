from benchmarks.learning_fix.predict import Predictor
from nmt.model.transformer.model import build_model
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from benchmarks.learning_fix.preprocess import dataset_generation, generated_iter_dataset
from tqdm import tqdm
from nmt.utils.context import Context
import os

import warnings
warnings.filterwarnings('ignore')

class Evaluator:
    def __init__(self, pred=None, save_filepath=None):

        self.predictor = pred
        self.save_filepath = save_filepath

    def evaluate_dataset(self, test_dataset):
        def tokenize(x):
            return x.split()

        predictions = []
        for source, _, target, _ in tqdm(test_dataset):
            prediction = self.predictor.predict_one(source=source, num_candidates=1)[0]
            predictions.append(prediction)

        hypotheses = [tokenize(prediction) for prediction in predictions]
        list_of_references = [[tokenize(target)] for source, _, target, _ in test_dataset]
        smoothing_function = SmoothingFunction()

        with open(self.save_filepath, 'w') as file:
            for (source, _, target, _), prediction, hypothesis, references in zip(test_dataset, predictions,
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

    data_source_type = "small"
    logger.info(f"Loading {data_source_type} data from disk and parse it as a bunch of batches ...")
    train_datasets, eval_datasets, test_datasets = dataset_generation(context, data_type=data_source_type)

    logger.info('Constructing dictionaries...')
    source_dictionary = train_datasets.src_vocab
    target_dictionary = train_datasets.tgt_vocab

    logger.info('Building model...')
    model = build_model(context, len(source_dictionary), len(target_dictionary))

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
    bleu_score = evaluator.evaluate_dataset(test_datasets)
    logger.info('Evaluation time : %d s', (datetime.now() - timestamp).seconds)
    logger.info("BLEU score : %f ", bleu_score)