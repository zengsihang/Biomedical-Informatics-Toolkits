from tqdm.auto import tqdm
import itertools
import random
import argparse
import os

# # Arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--umls_dir', type=str, default='umls/', help='input file')
# # args
# args = parser.parse_args()

def load_umls(args):
    with open(os.path.join(args.umls_dir, 'MRCONSO.RRF'), 'r') as f:
        lines = f.readlines()
    print(len(lines))

    cleaned = []
    count = 0
    for l in tqdm(lines):
        lst = l.rstrip("\n").split("|")
        cui, lang, synonym = lst[0], lst[1], lst[14]
        if lang != "ENG": continue # comment this out if you need all languages
        row = cui+"||"+synonym.lower()
        cleaned.append(row)
        # if len(cleaned)>1000:
            # break
    print (len(cleaned))

    print (len(cleaned))
    cleaned = list(set(cleaned)) 
    print (len(cleaned))

    print(cleaned[:3])

    umls_cui2namelist_dict = {} # constrauct cui to list of name dict, again
    idx2cui = {}
    idx2name = {}
    for idx, line in tqdm(enumerate(cleaned)):
        cui, name = line.split("||")
        if cui in umls_cui2namelist_dict:
            umls_cui2namelist_dict[cui].append(name)
        else:
            umls_cui2namelist_dict[cui] = [name]
        idx2cui[idx] = cui
        idx2name[idx] = name
    return umls_cui2namelist_dict, idx2cui, idx2name
