HF_TOKEN=UR_HF_TOKEN

export HF_ENDPOINT="https://hf-mirror.com"

# huggingface-cli download THUDM/chatglm3-6b --local-dir models/THUDM/chatglm3-6b --resume-download
huggingface-cli download google/gemma-2-2b --local-dir models/google/gemma-2-2b --token $HF_TOKEN --resume-download
huggingface-cli download google/gemma-2-2b-it --local-dir models/google/gemma-2-2b-it --token $HF_TOKEN --resume-download
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/meta-llama/Llama-3.2-1B --token $HF_TOKEN --resume-download
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir models/meta-llama/Llama-3.2-1B-Instruct --token $HF_TOKEN --resume-download

huggingface-cli download Abirate/english_quotes --local-dir datasets/Abirate/english_quotes --resume-download --repo-type dataset