#!/bin/sh

sbatch --time=168:00:00 \
       --partition=gpu \
       --gres=gpu:k80:1 \
       --cpus-per-task=8 \
       --mem=200g \
       --error=log.err \
       --output=log.out \
       --job-name="model_def" \
       ../keras python phase_two_definition.py
