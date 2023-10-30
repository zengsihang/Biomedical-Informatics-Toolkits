python mapping.py --query_path ../../all_traits.txt \
                  --output_path ../../all_traits_mapping_bafore_integrate.csv \
                  --umls_dir ~/umls \
                  --use_gpu_index 0 \
                  --k 10 \
                  --load_from_pool \
                  --model_name_or_path GanjinZero/coder_eng_pp # default: coderpp 
