# Is Temperature the Creativity for Large Language Models?

This repository contains the supplementary material / appendix to go with the paper “Is Temperature the Creativity for Large Language Models” by Max Peeperkorn, Tom Kouwenhoven, Dan Brown, and Anna Jordanous. 

The paper will appear at the [15th International Conference on Computational Creativity (ICCC)](https://computationalcreativity.net/iccc24/) held in Jönköping, Sweden from 17 to 21 June 2024. It contains the code, statistical analysis, data, and generated stories, and appendix with more details regarding the survey and stories.

## Contents

The appendix of the paper can be found in the `supplementary_materials.pdf`. This document contains the survey questions, definitions, the exemplar story and two other examples.

In the `data` folder, you will find the stories used in the survey in plain text format (and a metadata file), the survey results by participant and by evaluation, and all the stories generated for each temperature value (100 for each).

In the `scripts` folder, you will find the code we used to generated the stories and compute the embeddings.

In the `analysis.ipynb` notebook you will find the statistical analysis and the code that generated the figures in the paper.

## How to run the code?

Ensure that you have some version of [Llama 2 Chat](https://github.com/Meta-Llama/llama) downloaded. This project uses [llama.cpp](https://github.com/ggerganov/llama.cpp), you will need to convert model to `.guff` format (and perhaps quantise if necessary, we opted for `q6_k` setting). The script expects the following folder structure: `models/llama-2-70b-chat/ggml-model-f16.gguf`. 

```bash
python scripts/temperatures.py --experiment_name temperatures --model_name llama-2-70b-chat --n 100 \ 
    --temp_min 0.001 --temp_max 2.0 --temp_n 7 --temp_scale "lin" --prompt "Write a story."
```

When computing the embeddings, enter the model name and experiment output file you want to process, it will create a new pickle that includes the embedding vectors.

```bash
python scripts/compute_embeddings.py --experiment_path "output/temperatures.pickle" --model_name llama-2-70b-chat  
```

## Cite

```latex
@inproceedings{peeperkorn-etal-2024,
  title        = {Is Temperature the Creativity Parameter of Large Language Models?},
  author       = {Max Peeperkorn and Tom Kouwenhoven and Dan Brown and Anna Jordanous},
  booktitle    = {15th International Conference on Computational Creativity},
  year         = {2024},
  organization = {Association for Computational Creativity}
}
```
