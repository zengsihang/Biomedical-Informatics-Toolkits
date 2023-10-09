python get_bert_embed.py --input_file ../../all_traits.txt \
                        --output_file ../../embeddings_coderpp.csv \
                        --output_mode csv \
                        --model_name_or_path GanjinZero/coder_eng_pp \
                        --device cuda:0 \
                        --batch_size 32 \
                        --normalize \
                        --summary_method MEAN
