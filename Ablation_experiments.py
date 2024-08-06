#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import torch
from fancy_einsum import einsum
import tqdm
from jaxtyping import Float
import os

from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)

def main(input_file: str, head: int, layer: int):
    # Load the pre-trained model
    model = HookedTransformer.from_pretrained("gpt2-small")
    
    added_file = '.csv'
    original_path = '/home/causal-lm/josh/CausalCircuits2/datasets/simple-causal-dataset/'
    full_file = original_path + input_file + added_file

    # Load DataFrame
    try:
        df = pd.read_csv(full_file)
        print(f"Successfully loaded data from {full_file}")
    except FileNotFoundError:
        print(f"File not found: {full_file}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Check DataFrame columns
    if 'original' not in df.columns:
        print("Error: 'original' column not found in the DataFrame")
        return

    sentences = df['original'].tolist()
    # print(f"Number of sentences to process: {len(sentences)}")
    logit_diffs_list = []  # create an empty list to store logit diffs

    # Define the ablation hook function
    def ablate_head_hook(z: Float[torch.Tensor, "batch pos head_index d_head"], hook):
        z[:, -1, head, :] = 0
        return z

    # Process each sentence
    for i, sentence in enumerate(tqdm.tqdm(sentences)):
        try:
            tokens = model.to_tokens(sentence)
            pre_ablate_logits, pre_ablate_cache = model.run_with_cache(tokens)

            # Add the ablation hook
            model.blocks[layer].attn.hook_z.add_hook(ablate_head_hook)
            ablated_logits, ablated_cache = model.run_with_cache(tokens)

            # Convert logits from GPU to CPU and then to NumPy
            pre_ablate_logits = pre_ablate_logits.cpu().numpy()
            ablated_logits = ablated_logits.cpu().numpy()

            # Calculate logit differences
            logit_diffs = ablated_logits - pre_ablate_logits
            logit_diffs_list.append(logit_diffs)
            
            # if i % 100 == 0:
            #   print(f"Processed {i} sentences")

        except Exception as e:
            print(f"Error processing sentence {i}: {e}")
            logit_diffs_list.append(None)  # Append None for failed sentences

    # Add logit differences to the DataFrame
    df['logit_diffs'] = logit_diffs_list

    print("Logit differences calculated")

    new_path = "/mnt/isabelle-data/causal-circuits-josh/"
    added_name = '_zero_ablation.pkl'
    pickle_file = os.path.join(new_path, f"{input_file}_L{layer}H{head}{added_name}")
    print(pickle_file)

# Save the DataFrame to the pickle file
    df.to_pickle(pickle_file)


    # Load the DataFrame from the pickle file to verify
    try:
        loaded_df = pd.read_pickle(pickle_file)
        print("Loaded DataFrame from pickle:")
        print(loaded_df)
    except Exception as e:
        print(f"Error loading pickle file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and ablate head in GPT-2 model")
    parser.add_argument('input_file', type=str, help="Input CSV file path")
    parser.add_argument('--head', type=int, default=5, help="Head index to ablate")
    parser.add_argument('--layer', type=int, default=0, help="Layer index to ablate")
    args = parser.parse_args()

    main(args.input_file, args.head, args.layer)