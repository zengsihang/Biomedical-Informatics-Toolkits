from tqdm import tqdm
import torch
import numpy as np

def get_bert_embed(phrase_list, m, tok, args, normalize=True,\
        summary_method="CLS", tqdm_bar=True, batch_size=64):
    '''
    This function is used to generate embedding vectors for phrases in phrase_list
    
    param:
        phrase_list: list of phrases to be embeded
        m: model
        tok: tokenizer
        args: mainly args.device
        normalize: normalize the embeddings or not
        summary_method: method for generating embeddings from bert output, CLS for class token or MEAN for mean pooling
        tqdm_bar: progress bar
        batch_size: batch size for bert

    return:
        embeddings in numpy array with shape (phrase_list_length, embedding_dim)
    '''
    m = m.to(args.device)
    input_ids = []
    for phrase in tqdm(phrase_list):
        input_ids.append(tok.encode_plus(
            phrase, max_length=512, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
        # print(len(input_ids))
    m.eval()

    count = len(input_ids)
    now_count = 0
    output_list = []
    with torch.no_grad():
        if tqdm_bar:
            pbar = tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                now_count + batch_size, count)]).to(args.device)
            if summary_method == "CLS":
                embed = m(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(m(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            if now_count % 1000000 == 0:
                if now_count != 0:
                    output_list.append(output.cpu().numpy())
                    del output
                    torch.cuda.empty_cache()
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            if tqdm_bar:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
            del input_gpu_0
            torch.cuda.empty_cache()
        if tqdm_bar:
            pbar.close()
    output_list.append(output.cpu().numpy())
    del output
    torch.cuda.empty_cache()
    # print('end')
    return np.concatenate(output_list, axis=0)
