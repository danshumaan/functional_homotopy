# Functional Homotopy for AutoDAN

### Setup:
Make the following updates in `run_fhautodan.sh`
1. Update the `TEMPLATE` field to either `{llama-2/llama-3/mistral/vicuna_v1.1}`, depending on the model you wish to run.
2. Update `BASE_MODEL_PATH` with the model path.
3. Update `FINETUNED_MODEL_PATH` to the folder with the LoRA adapters obtained by finetuning.
4. `JUDGE_PATH` should point to the [HarmBench judge model](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls).
5. Update `TAG` for save folder names.

### Running Baseline AutoDAN:
Update the `STEPS` field in `run_fhgcg.sh` to 1000.

Both baselines use the same command:
```
bash run_fhautodan.sh 0 200 0 2 0
```
The command line is passing the first two arguments to the `START` and `END` fields in the `run_fhautodan.sh` file. You can change these to control the slice of the dataset being processed.

The last command line argument is passed to the `EPOCHS` field, which controls the initial checkpoint used in functional homotopy. '0' indicates the base model and will run AutoDAN, for the designated number of steps.

### Running FH-AutoDAN:
Update the `STEPS` field in `run_fhgcg.sh` to 200.

We simply modify the previous command:
```
bash run_fhautodan.sh 0 200 0 2 500
```
The command line is passing the first two arguments to the `START` and `END` fields in the `run_fhautodan.sh` file. You can change these to control the slice of the dataset being processed.

Here, the last command line argument is passed to the `EPOCHS` field, which controls the initial checkpoint used in functional homotopy. Setting this as '500' will start the binary search process where no checkpoint will take more than 200 iterations to process.