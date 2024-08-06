#!/bin/bash

# Define input file
input_file="aos_reviewed"

# List of layer and head combinations
combinations=(
  "10 0"
  "9 0"
  "11 3"
  "8 8"
  "6 7"
)

# Loop through each combination
for combination in "${combinations[@]}"; do
  IFS=' ' read -r layer head <<< "$combination"
  echo "Running program with layer=$layer and head=$head"
  /home/causal-lm/miniconda3/bin/python /home/causal-lm/josh/CausalCircuits2/scripts/Ablation_experiments.py "$input_file" --head "$head" --layer "$layer"
done

# Specific combinations based on provided syntax
declare -A special_combinations=(
  ["0"]="0 2 6 8 9 10 11"
  ["1"]="1 2 3 4 5 6 7 8 9 10"
  ["2"]="1 5 7 10"
)

for layer in "${!special_combinations[@]}"; do
  heads=${special_combinations[$layer]}
  for head in $heads; do
    echo "Running program with layer=$layer and head=$head"
    /home/causal-lm/miniconda3/bin/python /home/causal-lm/josh/CausalCircuits2/scripts/Ablation_experiments.py "$input_file" --head "$head" --layer "$layer"
  done
done
