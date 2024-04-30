#
#   Created by Max Peeperkorn on 20/12/2023.
#
#   This script runs the experiment to investigate the different modes to instruct Llama 2 to generate stories.
#

import os
import pickle
import fire

from llama_cpp import Llama
from tqdm import tqdm


def read_pickle(file_path: str) -> list:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def main(experiment_path: str, model_name="llama-2-7b-chat") -> int:
    experiment_name = os.path.splitext(os.path.split(experiment_path)[-1])[0]
    print(f"Running embedding step for experiment {experiment_name}")
    
    output_file_path = os.path.join(os.getcwd(), "output", f"{experiment_name}+embeddings.pickle")
    if not os.path.exists(os.path.join(os.getcwd(), "output")):
        os.makedirs(os.path.join(os.getcwd(), "output"))

    print(f"Loading {model_name}", end="...\n")

    model_path = os.path.join("models", model_name, "ggml-model-f16.gguf")
    if not os.path.exists(model_path):
        raise Exception("Model not found.")

    llm_embed = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096, embedding=True, verbose=True)

    data_path = os.path.join(experiment_path)
    if not os.path.exists(model_path):
        raise Exception("Pickle not found.")

    data = read_pickle(data_path)
    results = []
    for item in tqdm(data, desc="Calculating embeddings"):
        embedding = llm_embed.embed(item['story'])
        result = {
            **item, "embedding": embedding
        }
        results.append(result)

    print("\nSaving results.")
    # -- save experimental output
    with open(output_file_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")
    return 0


if __name__ == "__main__":
    fire.Fire(main)
