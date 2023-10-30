python get_bert_embed.py --input_file ../../two_phrases.txt \
                        --output_file ../../two_embedding_coder.csv \
                        --output_mode csv \
                        --model_name_or_path GanjinZero/coder_eng \
                        --device cuda:0 \
                        --batch_size 32 \
                        --normalize \
                        --summary_method CLS
