# TREDNet_v1.1 Setup Instructions

This project requires a specific Conda environment to be created and activated before running the pipeline.

## Step 1. Create the Conda Environment
Make sure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

Create the environment using the provided YAML file:

```bash
conda env create -f tf2.15-gpu.yml
```

## Step 2. Activate the Environment

```bash
conda activate tf2.15-gpu
```

## Step 3. Run the Testing Script

Once the environment is active, run the testing script:

```bash
bash submit_testing.sh
```
