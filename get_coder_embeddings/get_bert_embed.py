from tqdm import tqdm
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModel
import pandas as pd

def get_bert_embed(phrase_list, m, tok, device, normalize=True, \
        summary_method="CLS", batch_size=64):
    '''
    This function is used to generate embedding vectors for phrases in phrase_list
    
    param:
        phrase_list: list of phrases to be embeded
        m: model
        tok: tokenizer
        device: device for inference
        normalize: normalize the embeddings or not
        summary_method: method for generating embeddings from bert output, CLS for class token or MEAN for mean pooling
        tqdm_bar: progress bar
        batch_size: batch size for bert

    return:
        embeddings in numpy array with shape (phrase_list_length, embedding_dim)
    '''
    m = m.to(device)
    input_ids = []
    for phrase in tqdm(phrase_list):
        input_ids.append(tok.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
        # print(len(input_ids))
    m.eval()

    count = len(input_ids)
    now_count = 0
    output_list = []
    with torch.no_grad():
        pbar = tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                now_count + batch_size, count)]).to(device)
            if summary_method == "CLS":
                embed = m(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(m(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            if now_count % 1000000 < batch_size:
                if now_count != 0:
                    output_list.append(output.cpu().numpy())
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
        pbar.close()
    output_list.append(output.cpu().numpy())
    # print('end')
    return np.concatenate(output_list, axis=0)

def run(args):
    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = args.device
    phrase_list = []
    with open(args.input_file, 'r') as f:
        for line in f:
            phrase_list.append(line.strip())
    embed = get_bert_embed(phrase_list, model, tokenizer, device, args.normalize, args.summary_method, args.batch_size)
    if args.output_mode == 'npy':
        np.save(args.output_file, embed)
    elif args.output_mode == 'csv':
        # phrase list in the first column
        embed = np.concatenate((np.array(phrase_list).reshape(-1, 1), embed), axis=1)
        header = ['phrase'] + ['dim' + str(i) for i in range(embed.shape[1]-1)]
        # make it a pd dataframe
        embed = pd.DataFrame(embed)
        # write to csv
        embed.to_csv(args.output_file, index=False, header=header)
    else:
        raise ValueError('output mode not supported')
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='data/phrase_list.txt', help='input file containing phrases in each row')
    parser.add_argument('--output_file', type=str, default='data/phrase_embed.npy', help='output file for saving embeddings')
    parser.add_argument('--output_mode', type=str, default='npy', help='output mode for saving embeddings, chosen from npy and csv')
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased', help='path to pretrained bert model (in huggingface format)')
    parser.add_argument('--device', type=str, default='cuda', help='device for inference')
    parser.add_argument('--summary_method', type=str, default='CLS', help='CLS or MEAN')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--normalize', action='store_true', help='normalize the embeddings or not')
    args = parser.parse_args()

    run(args)


