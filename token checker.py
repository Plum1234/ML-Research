import pandas as pd
from transformers import GPT2Tokenizer
import mwparserfromhell
import nltk
from datasets import load_dataset

# Load the CSV file
file_path = 'dataset.csv'
data = pd.read_csv(file_path)

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenize the first sentence
first_sentence = data['original'][0]
first_sentence_tokens = tokenizer.encode(first_sentence)

# Get the token length of the first sentence
first_sentence_length = len(first_sentence_tokens)
print(f"First sentence token length: {first_sentence_length}")

# List to store sentences that do not match the first sentence token length
inconsistent_sentences = []

# Check each sentence in the dataset
for sentence in data['original']:
    sentence_tokens = tokenizer.encode(sentence)
    if len(sentence_tokens) != first_sentence_length:
        inconsistent_sentences.append(sentence)

# Print out the inconsistent sentences
if inconsistent_sentences:
    print("Sentences that do not match the token length of the first sentence:")
    for sentence in inconsistent_sentences:
        print(sentence)
else:
    print("All sentences have the same token length.")