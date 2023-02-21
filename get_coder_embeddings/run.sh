bash get_bert_embed --input_file phrase.txt \
                    --output_file embeddings.npy \
                    --output_mode npy \
                    --model_name_or_path GanjinZero/coder_eng_pp \
                    --device cuda:0 \
                    --batch_size 32 \
                    --normalize \
                    --summary_method CLS
