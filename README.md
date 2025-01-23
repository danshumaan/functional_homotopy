# Functional Homotopy
Code submission for ICLR 2025

## Prerequisite
We first create a python 3.11.4 virtual environment and install the packages in `requirements.txt`

After installing `nltk`, use the NLTK downloader for the following:
```
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('wordnet')
>>> nltk.download('punkt')
```

## Running Baselines and Functional Homotopy
Run the appropriate bash script for a given model and method, as described in `fhgr` and `fhautodan`. 

You can modify the GPU device, LoRA adapter path, base model path, judge model path, dataset slices etc. in the `.sh` scripts found in the folder. 

Running either scripts found in the folder will save a json file in the appropriate path. We provide a sample output (`sample_output.json`) for a single prompt, run for Llama-2 7B Chat. 

Using binary search, we inspect checkpoints **250 -> 125 -> 62 -> 31 -> 15 -> 7 -> 3 -> 1 -> Baseline**, which is the original model. In this example, we successfully jailbreak all checkpoints.

### Save file format:
1. `completions`: is a list with the completions to adversarial inputs, obtained at the end of every jailbreaking attempt for a checkpoint.
2. `tasks`: is a list of harmful instructions obtained from AdvBench and HarmBench.
3. `goals`: is a list of affirmative responses (*"Sure, here is..."*)
4. `iterations`: is a list of lists, containing the total iterations (iteration index) taken per checkpoint to find a jailbreak for every task.
5. `losses`: is a list of nested lists. Each list element tracks the checkpoint losses for a given prompt. Each element of the nested list contains the losses for individual checkpoints as the suffix is optimized.
6. `adv_strings`: is similarly a list of nested lists. Each list element corresponds to the suffixes found by GR, GCG or AutoDAN for a given prompt. Each nested list contains the adversarial suffixes that either succeeded or failed at the end of available iterations for each checkpoint.
7. `successes`: is a list of lists, where each element is a list of a successful or failed jailbreak attempt for that checkpoint index. 

You can change which checkpoints are being used by modifying the scripts. 
