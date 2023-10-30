'''
Mapping query terms to cuis
'''

from find_knn import find_knn
from get_bert_embed import get_bert_embed
from load_umls import load_umls
import pandas as pd
import numpy as np
import argparse
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm, trange
import pickle

def load_query(query_path):
    '''
    This function is used to load query terms from file.
    Default is a csv file with the first column as query terms.
    You can design your own load_query function to load your own query terms.
    Return: list of query terms [term1, term2, ...]
    '''
    # the first column is the query term, returns a list of query terms
    # return pd.read_csv(query_path, header=None)[0].tolist()
    with open(query_path, 'r') as f:
        return [line.strip() for line in f]

def save_npy(data, path):
     '''
     This function is used to save numpy array to file.
     '''
     with open(path, 'wb') as f:
         np.save(f, data)

def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # use argparse to control inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', type=str, default='data/query.csv', help='path to query file')
    parser.add_argument('--umls_dir', type=str, default='umls/', help='path to umls directory')
    parser.add_argument('--output_path', type=str, default='data/query_cui.csv', help='path to output file')
    parser.add_argument('--use_gpu_index', type=int, default=0, help='gpu index in use')
    parser.add_argument('--k', type=int, default=10, help='number of nearest neighbors')
    parser.add_argument('--model_name_or_path', type=str, default='GanjinZero/coder_eng_pp', help='model name or path')
    parser.add_argument('--load_from_pool', action='store_true', help='whether to load pool from file')
    parser.add_argument('--use_multiple_gpu', action='store_true', help='whether to use multiple gpu for faiss')
    parser.add_argument('--exact', action='store_true', help='whether to use exact search')
    
    args = parser.parse_args()
    args.device = 'cuda:{}'.format(args.use_gpu_index)

    # load query terms
    print('Loading query terms...')
    query_terms = load_query(args.query_path)

    # get bert embedding
    print('Getting bert embedding...')
    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    query = get_bert_embed(query_terms, model, tokenizer, summary_method='MEAN', device=args.device)
    if args.load_from_pool:
        print('Loading umls from file...')
        idx2cui = read_pkl('idx2cui.pkl')
        idx2term = read_pkl('idx2term.pkl')
        pool = np.load('umls_pool_embedding.npy')
    else:
        # load umls
        print('Loading umls...')
        umls_cui2termlist_dict, idx2cui, idx2term = load_umls(args.umls_dir)
        umls_term_list = list(idx2term.values())
        pool = get_bert_embed(umls_term_list, model, tokenizer, summary_method='MEAN', device=args.device)
        ######### save dicts and embeddings to save time for next time
        save_npy(pool, 'umls_pool_embedding.npy')
        save_pkl(idx2cui, 'idx2cui.pkl')
        save_pkl(idx2term, 'idx2term.pkl')
    
    # find nearest neighbors
    print('Finding nearest neighbors...')
    similarity, indices = find_knn(pool, query, use_gpu_index=args.use_gpu_index, k=args.k, use_multi_gpu=args.use_multiple_gpu, exact=args.exact)

    # map k nearest neighbors to cuis and terms for each query term
    print('Mapping nearest neighbors to cuis and terms...')
    rows = []
    for i in trange(len(query_terms)):
        rows.append([query_terms[i]])
        map_cuis = [idx2cui[int(idx)] for idx in indices[i]]
        map_terms = [idx2term[int(idx)] for idx in indices[i]]
        map_sim = [str(round(sim, 4)) for sim in similarity[i]]
        map_pairs = list(zip(map_cuis, map_terms, map_sim))
        rows[-1] += ['||'.join(pair) for pair in map_pairs]
    
    # save results to csv file
    print('Saving results to csv file...')
    df = pd.DataFrame(rows)
    header = ['query'] + ['cui{}||term{}||sim{}'.format(i, i, i) for i in range(args.k)]
    df.to_csv(args.output_path, index=False, header=header)



