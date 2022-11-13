# from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import pickle
import sys
sys.path.append('..')
sys.path.append('.')

import json
import time
import copy
import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from pprint import pprint

from trec_eval import trec_eval
from dense_retrievers import load_dense_retriever
from utils import check_dir_exist_or_build, json_dumps_arguments, set_seed, get_has_qrel_label_sample_ids



def build_faiss_index(args):
    logger.info("Building Faiss Index...")
    # ngpu = faiss.get_num_gpus()
    ngpu = args.n_gpu_for_faiss
    gpu_resources = []
    tempmem = -1

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    cpu_index = faiss.IndexFlatIP(768)  
    index = None
    if args.use_gpu_in_faiss:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres,
                                                    vdev,
                                                    cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index

    return index

def get_query_embs(args):
    query_tokenizer, query_encoder = load_dense_retriever("ANCE", "query", args.retriever_path)
    query_encoder = query_encoder.to(args.device)
    
    with open(args.eval_file_path, "r") as f:
        data = json.load(f)
    
    query_encoding_dataset = []
    for record in data:
        sample_id = record['sample_id']
        query = record[args.eval_field_name]
        query_encoding_dataset.append([sample_id, query])

    def query_encoding_collate_fn(batch):
        bt_sample_ids, bt_src_seq = list(zip(*batch)) # unzip
        bt_src_encoding = query_tokenizer(bt_src_seq, 
                                          padding="longest", 
                                          max_length=args.max_query_length, 
                                          truncation=True, 
                                          return_tensors="pt")
        bt_input_ids, bt_attention_mask = bt_src_encoding.input_ids, bt_src_encoding.attention_mask
        return {"bt_sample_ids": bt_sample_ids, 
                "bt_input_ids":bt_input_ids, 
                "bt_attention_mask":bt_attention_mask}

    test_loader = DataLoader(query_encoding_dataset, 
                             batch_size = 32, 
                             shuffle=False, 
                             collate_fn=query_encoding_collate_fn)

    query_encoder.zero_grad()
    embeddings = []
    eid2sid = []    # embedding idx to sample id
    has_qrel_label_sample_ids = get_has_qrel_label_sample_ids(args.qrel_file_path)
    with torch.no_grad():
        for batch in tqdm(test_loader):
            query_encoder.eval()
            bt_sample_ids = batch["bt_sample_ids"]
            bt_input_ids = batch['bt_input_ids'].to(args.device)
            bt_attention_mask = batch['bt_attention_mask'].to(args.device)

            query_embs = query_encoder(bt_input_ids, bt_attention_mask)
            query_embs = query_embs.detach().cpu().numpy()

            sifted_sample_ids = []
            sifted_query_embs = []
            for i in range(len(bt_sample_ids)):
                if bt_sample_ids[i] not in has_qrel_label_sample_ids:
                    continue
                sifted_sample_ids.append(bt_sample_ids[i])
                sifted_query_embs.append(query_embs[i].reshape(1, -1))
        
            if len(sifted_query_embs) > 0:
                sifted_query_embs = np.concatenate(sifted_query_embs)

                embeddings.append(sifted_query_embs)
                eid2sid.extend(sifted_sample_ids)
            else:
                continue

        embeddings = np.concatenate(embeddings, axis = 0)
    
    torch.cuda.empty_cache()
    return embeddings, eid2sid


def faiss_flat_retrieval_one_by_one_and_finally_merge(args, query_embs):
    index = build_faiss_index(args)

    merged_candidate_matrix = None
    # Automaticall get the number of doc blocks
    for filename in os.listdir(args.index_path):
        try:
            args.num_doc_block = int(filename.split(".")[1]) + 1
        except:
            continue
    print("Automatically detect that the number of doc blocks is: {}".format(args.num_doc_block))
    
    for block_id in range(args.num_doc_block):
        logger.info("Loading doc block " + str(block_id))

        # load doc embeddings
        with open(os.path.join(args.index_path, "doc_emb_block.{}.pb".format(block_id)), 'rb') as handle:
            cur_doc_embs = pickle.load(handle)
        with open(os.path.join(args.index_path, "doc_embid_block.{}.pb".format(block_id)), 'rb') as handle:
            cur_eid2did = pickle.load(handle)
            if isinstance(cur_eid2did, list):
                cur_eid2did = np.array(cur_eid2did)

        # Split to avoid the doc embeddings to be too large
        num_total_doc_per_block = len(cur_doc_embs)
        num_doc_per_split = 10000000    # please set it according to your GPU size. 700w doc needs ~28GB
        num_split_block = max(1, num_total_doc_per_block // num_doc_per_split)
        logger.info("num_total_doc: {}".format(num_total_doc_per_block))
        logger.info("num_doc_per_split: {}".format(num_doc_per_split))
        logger.info("num_split_block: {}".format(num_split_block))
        cur_doc_embs_list = np.array_split(cur_doc_embs, num_split_block)
        cur_eid2did_list = np.array_split(cur_eid2did, num_split_block)
        for split_idx in range(len(cur_doc_embs_list)):
            cur_doc_embs = cur_doc_embs_list[split_idx]
            cur_eid2did = cur_eid2did_list[split_idx]
            logger.info("Adding block {} split {} into index...".format(block_id, split_idx))
            index.add(cur_doc_embs)
            
            # ann search
            tb = time.time()
            D, I = index.search(query_embs, args.top_n)
            elapse = time.time() - tb
            logger.info({
                'time cost': elapse,
                'query num': query_embs.shape[0],
                'time cost per query': elapse / query_embs.shape[0]
            })

            candidate_did_matrix = cur_eid2did[I] # doc embedding_idx -> real doc id
            D = D.tolist()
            candidate_did_matrix = candidate_did_matrix.tolist()
            candidate_matrix = []

            for score_list, doc_list in zip(D, candidate_did_matrix):
                candidate_matrix.append([])
                for score, doc in zip(score_list, doc_list):
                    candidate_matrix[-1].append((score, doc))
                assert len(candidate_matrix[-1]) == len(doc_list)
            assert len(candidate_matrix) == I.shape[0]

            index.reset()
            del cur_doc_embs
            del cur_eid2did

            if merged_candidate_matrix == None:
                merged_candidate_matrix = candidate_matrix
                continue
            
            # Merge
            merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
            merged_candidate_matrix = []
            for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                            candidate_matrix):
                p1, p2 = 0, 0
                merged_candidate_matrix.append([])
                while p1 < args.top_n and p2 < args.top_n:
                    if merged_list[p1][0] >= cur_list[p2][0]:
                        merged_candidate_matrix[-1].append(merged_list[p1])
                        p1 += 1
                    else:
                        merged_candidate_matrix[-1].append(cur_list[p2])
                        p2 += 1
                while p1 < args.top_n:
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                while p2 < args.top_n:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1

    merged_D, merged_I = [], []

    for merged_list in merged_candidate_matrix:
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list:
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)

    logger.info(merged_D.shape)
    logger.info(merged_I.shape)

    return merged_D, merged_I

def dense_retrieval(args):
    query_embs, eid2sid = get_query_embs(args)
    score_mat, did_mat = faiss_flat_retrieval_one_by_one_and_finally_merge(args, query_embs)
    
    # write to file
    run_trec_file = os.path.join(args.retrieval_output_path, "res.trec")
    with open(run_trec_file, "w") as f:
        for eid in range(len(did_mat)):
            sample_id = eid2sid[eid]
            retrieval_scores = score_mat[eid]
            retrieved_dids = did_mat[eid]
            for i in range(len(retrieval_scores)):
                rank = i + 1
                doc_id = retrieved_dids[i]
                rank_score = args.top_n - i # use the rank score for pytrec
                real_score = retrieval_scores[i] 
                f.write("{} {} {} {} {} {} {}".format(sample_id, "Q0", doc_id, rank, rank_score, real_score, "ance"))
                f.write('\n')
            
    # evaluation
    trec_eval(run_trec_file, args.qrel_file_path, args.retrieval_output_path, args.rel_threshold)
    


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_file_path", type=str, required=True)
    parser.add_argument("--eval_field_name", type=str, required=True, help="Field name of the rewrite in the eval_file. E.g., t5_rewrite")
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--qrel_file_path", type=str, required=True)
    parser.add_argument("--retriever_path", type=str, required=True)
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")

    parser.add_argument("--use_gpu_in_faiss", action="store_true", help="whether to use gpu in faiss or not.")
    parser.add_argument("--n_gpu_for_faiss", type=int, default=1, help="should be set if use_gpu_in_faiss")
    
    
    parser.add_argument("--top_n", type=int, default=1000)
    parser.add_argument("--rel_threshold", type=int, required=True, help="CAsT-20: 2, Others: 1")

    parser.add_argument("--retrieval_output_path", type=str, required=True)
    parser.add_argument("--force_emptying_dir", action="store_true", help="Force to empty the (output) dir.")
    parser.add_argument("--seed", type=int, default=42)

    # main
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    check_dir_exist_or_build([args.retrieval_output_path], args.force_emptying_dir)
    json_dumps_arguments(os.path.join(args.retrieval_output_path, "parameters.txt"), args)
 
    logger.info("---------------------The arguments are:---------------------")
    pprint(args)

    return args



if __name__ == '__main__':
    args = get_args()
    set_seed(args) 
    dense_retrieval(args)
    logger.info("Dense retrieval and evaluation finish!")