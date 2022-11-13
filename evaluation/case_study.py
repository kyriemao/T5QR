from IPython import embed
import json
import pytrec_eval

def load_run_trec(run_trec_file):
    with open(run_trec_file, 'r' )as f:
        run_data = f.readlines()
    runs = {}
    for line in run_data:
        line = line.split(" ")
        sample_id = line[0]
        doc_id = line[2]
        score = float(line[4])
        if sample_id not in runs:
            runs[sample_id] = {}
        runs[sample_id][doc_id] = score

    return runs

def load_qrel_trec(qrel_trec_file, rel_threshold):
    with open(qrel_trec_file, 'r') as f:
        qrel_data = f.readlines()
    qrels = {}
    qrels_ndcg = {}
    for line in qrel_data:
        line = line.strip().split("\t")
        query = line[0]
        doc_id = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][doc_id] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][doc_id] = rel
    return qrels, qrels_ndcg



def output_case_study_file(res1, res2, tag1, tag2, metric_name, orig_eval_file_path, case_study_file_path):
    with open(orig_eval_file_path, "r") as f:
        data = json.load(f)
    
    outputs = []
    for record in data:
        sample_id = record['sample_id']
        if sample_id not in res1:
            continue
        val1 = res1[sample_id][metric_name]
        val2 = res2[sample_id][metric_name]
        record["{}_val1"] = val1
        record["{}_val2"] = val2
        if val1 > val2:
            record["which_is_better"] = tag1
        elif val1 < val2:
            record["which_is_better"] = tag2 
        else:
            record["which_is_better"] = "same"       
        outputs.append(record)
    
    with open(case_study_file_path, "w") as f:
        f.write(json.dumps(outputs, indent=4))

    print("case study output file ok!")

def which_is_better(run_trec_file1, run_trec_file2, tag1, tag2, metric_name, qrel_trec_file, rel_threshold, orig_eval_file_path, case_study_file_path):
    runs1 = load_run_trec(run_trec_file1)
    runs2 = load_run_trec(run_trec_file2)
    qrels, qrels_ndcg = load_qrel_trec(qrel_trec_file, rel_threshold)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res1 = evaluator.evaluate(runs1)
    res2 = evaluator.evaluate(runs2)
    
    output_case_study_file(res1, res2, tag1, tag2, metric_name, orig_eval_file_path, case_study_file_path)
   

if __name__ == "__main__":
    run_trec_file1 = "./results/cast20_ance_t5_rewrite/res.trec"
    run_trec_file2 = "./results/cast20_ance_oracle_rewrite/res.trec"
    tag1 = "t5_rewrite"
    tag2 ="oracle_rewrite"
    metric_name = "ndcg_cut_3"
    qrel_trec_file = "/data1/kelong_mao/datasets/cast20/preprocessed/cast20_qrel.tsv"
    rel_threshold = 2
    orig_eval_file_path = "../outputs/t5_rewrite_cast20.json"
    case_study_file_path = "./results/case_study_cast20.json"
    which_is_better(run_trec_file1, run_trec_file2, tag1, tag2, metric_name, qrel_trec_file, rel_threshold, orig_eval_file_path, case_study_file_path)

   