
mkdir -p data
echo "Downloading data..."
wget -c https://raw.githubusercontent.com/AlibabaPAI/FlashModels/1c176f39a14de681656f96d8e77a9fa432dbf6cf/data/wikitext-2-raw-v1.json -O data/wikitext-2-raw-v1.json


mkdir -p models
cd models

model_urls=(
    "https://www.modelscope.cn/Qwen/Qwen2.5-3B-Instruct.git"
    "https://www.modelscope.cn/LLM-Research/Llama-3.2-3B-Instruct.git"
    "https://www.modelscope.cn/LLM-Research/gemma-2-2b-it.git"
)

clone_if_not_exists() {
    local url=$1
    local dir_name=$(basename $url .git)

    if [ ! -d "$dir_name" ]; then
        git clone $url
    else
        echo "Directory $dir_name already exists, skipping clone."
    fi
}

git lfs install

for url in "${model_urls[@]}"; do
    clone_if_not_exists $url
done

cd ..
