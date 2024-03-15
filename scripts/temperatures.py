#
#   Created by Max Peeperkorn on 29/12/2023.
#
#   This script runs the experiment to investigate the effects of temperature
#   when instructing Llama 2 to generate stories.
#

import fire
import numpy as np
import os
import pickle

from llama_cpp import Llama
from llama_cpp.llama_chat_format import ChatFormatterResponse
from tqdm import tqdm


B_INST, E_INST = '[INST]', '[/INST]'
STOP = ["user", "system"]
SUFFIX = "Here it is:\n\n"

# set the default provided by llama-cpp-python lib
SEED = 4294967295


def make_path(*args) -> str:
    path = os.path.join(*args)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def format_without_system(messages: list, suffix="") -> ChatFormatterResponse:
    """ custom message formatting, skips over any system prompts, and adds a suffix as a start for the completion. """
    prompt = ""
    for message in messages:
        if message['role'] == 'user':
            prompt += f"{B_INST}{message['content']}"

        if message['role'] == 'assistant':
            prompt += f"{E_INST}{message['content']}"

    prompt = prompt + E_INST + suffix
    return ChatFormatterResponse(prompt=prompt)


def experiment(llm: Llama, instruction: str, temperatures: np.ndarray, n: int) -> list:
    results = []
    for t in temperatures:
        print(f"Generating stories for temperature {t}.")

        for i in (progress_bar := tqdm(range(n))):
            progress_bar.set_description(f"{t:.3f}, #{i}")
            response = llm(instruction.prompt, max_tokens=4096, temperature=t, top_p=1.0, top_k=50,
                           repeat_penalty=1.0, logprobs=1, stop=STOP)
            result = {
                "i": i,
                "t": t,
                "prompt": instruction.prompt,
                "story": response['choices'][0]['text'],
                "output": response
            }
            results.append(result)
            llm.reset()
    return results


def main(experiment_name="temperatures-7b", model_name="llama-2-7b-chat", prompt="Write a story.",
         n=30, temp_scale="lin", temp_min=0.0, temp_max=2.0, temp_n=7, verbose=False) -> int:
    print(f"Running generation step for experiment {experiment_name}")
    print(f"Using seed {SEED}")

    output_path = make_path(os.getcwd(), "output")
    output_file_path = os.path.join(output_path, f"{experiment_name}.pickle")

    print(f"Loading {model_name}", end="...\n")

    model_path = os.path.join("models", model_name, "ggml-model-f16.gguf")
    if not os.path.exists(model_path):
        raise Exception("Model not found.")

    llm = Llama(model_path=model_path, n_gpu_layers=-1, seed=SEED,
                n_ctx=4096, logits_all=True, verbose=verbose)

    messages = [{"role": "user", "content": prompt}]
    instruction = format_without_system(messages, SUFFIX)

    print("Instruction:")
    print(instruction)

    if temp_scale == "lin":
        temperatures = np.linspace(temp_min, temp_max, temp_n)
    else:
        temperatures = np.logspace(
            np.log10(temp_min), np.log10(temp_max), temp_n)

    results = experiment(llm, instruction, temperatures, n)

    print("\nSaving results.")
    # -- save experimental output
    with open(output_file_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")
    return 0


if __name__ == "__main__":
    fire.Fire(main)
