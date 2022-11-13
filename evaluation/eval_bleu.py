from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('..')
sys.path.append('.')

import json
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def eval_bleu(args):
    with open(args.eval_file_path) as f:
        data = json.load(f)
    
    # chencherry = SmoothingFunction()
    bleu_list = []
    for record in data:
        reference = [record['oracle_rewrite'].split()]
        prediction = record[args.rewrite_field_name].split()
        bleu = sentence_bleu(reference, prediction)
        bleu_list.append(bleu)
                
    logger.info("Eval file: {}, Rewrite Field: {}".format(args.eval_file_path, args.eval_field_name))
    logger.info("BLEU Results:")
    logger.info(sum(bleu_list) / len(bleu_list))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file_path", type=str, required=True, help="Path of the file for rewriting evaluation.")
    parser.add_argument("--eval_field_name", type=str, required=True, help="Field name of the rewrite in the eval_file. E.g., t5_rewrite")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    eval_bleu(args)
