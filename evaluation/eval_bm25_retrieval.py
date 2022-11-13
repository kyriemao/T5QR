# from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import sys
# sys.path.append('..')
sys.path.append('.')

import json
import argparse
from pprint import pprint
from ..utils import check_dir_exist_or_build, json_dumps_arguments, set_seed

from pyserini.search.lucene import LuceneSearcher
from trec_eval import trec_eval

"""
To avoid memory boom.
"""
def auto_split_to_chunks(data):
    num_query_per_chunk = 1000
    num_chunk = max(1, len(data) // num_query_per_chunk)
    chunk_size = max(1, len(data) // num_chunk)
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i: i + chunk_size])
    return chunks

def bm25_retrieval(args):
    query_list = []
    qid_list = []

    with open(args.eval_file_path, "r") as f:
        data = json.load(f)

    for record in data:
        rewrite = record[args.rewrite_field_name]
        query_list.append(rewrite)
        qid_list.append(record['sample_id'])
   

    qid_chunks = auto_split_to_chunks(qid_list)
    query_chunks = auto_split_to_chunks(query_list)
    
    # pyserini search
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = []
    for chunk_id in range(len(qid_chunks)):
        cur_hits = searcher.batch_search(query_chunks[chunk_id], qid_chunks[chunk_id], k = args.top_n, threads = 20)
        hits.extend(cur_hits)
    
    # write to file
    run_trec_file = os.path.join(args.retrieval_output_path, "res.trec")
    with open(run_trec_file, "w") as f:
        for qid in hits:
            for i, item in enumerate(hits[qid]):
                rank = i + 1
                score = args.top_n - i
                doc_id = item.docid
                f.write("{} {} {} {} {} {}".format(qid, "Q0", doc_id, rank, score, "bm25"))

    
    # evaluation
    trec_eval(run_trec_file, args.qrel_file_path, args.retrieval_output_path, args.rel_threshold)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_file_path", type=str, required=True)
    parser.add_argument("--eval_field_name", type=str, required=True, help="Field name of the rewrite in the eval_file. E.g., t5_rewrite")
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--qrel_file_path", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=1000)
    
    parser.add_argument("--bm25_k1", type=float, required=True)   # 0.82 for qrecc, 0.9 for topiocqa
    parser.add_argument("--bm25_b", type=float, required=True)    # 0.68 for qrecc, 0.4 for topiocqa
    parser.add_argument("--rel_threshold", type=int, required=True, help="CAsT-20: 2, Others: 1")

    parser.add_argument("--retrieval_output_path", type=str, required=True)
    parser.add_argument("--force_emptying_dir", action="store_true", help="Force to empty the (output) dir.")

    # main
    args = parser.parse_args()
    check_dir_exist_or_build([args.retrieval_output_path])
    json_dumps_arguments(os.path.join(args.retrieval_output_path, "parameters.txt"), args)
 
    logger.info("---------------------The arguments are:---------------------")
    pprint(args)
    
    return args



if __name__ == '__main__':
    args = get_args()
    set_seed(args) 
    bm25_retrieval(args)
    logger.info("BM25 retrieval and evaluation finish!")