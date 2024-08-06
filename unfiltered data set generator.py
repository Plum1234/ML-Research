import pandas as pd
from transformers import GPT2Tokenizer
import mwparserfromhell
import nltk
from datasets import load_dataset

# Load the CSV file
file_path = 'alb_the_reviewed.csv'
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

wikipedia_dataset = load_dataset("wikipedia", "20220301.en", split='train', trust_remote_code=True)

print("LENGTH: ")
print(first_sentence_length)

# Function to check token length
def is_desired_length(text, target_length):
    tokens = tokenizer.encode(text, return_tensors='pt')
    if tokens.size(1) == target_length:
        print(tokens.size(1))
        print(text)
    return tokens.size(1) == target_length and tokens.size(1) <= 1024

# Filter sentences with the same token length as the first sentence
matching_sentences = []
for article in wikipedia_dataset:
    for sentence in article['text'].split('. '):
        sentence = sentence.strip()
        if is_desired_length(sentence, first_sentence_length):
            matching_sentences.append(sentence)
            if len(matching_sentences) >= 5000:
                break
    if len(matching_sentences) >= 5000:
        break

# Check if we have collected enough sentences
if len(matching_sentences) < 5000:
    print(f"Only found {len(matching_sentences)} sentences with the required token length.")
else:
    # Ensure that all sentences have the exact token length
    matching_sentences = matching_sentences[:5000]  # Truncate to exactly 5000 sentences

    # Create a new DataFrame with the required format
    new_dataset = pd.DataFrame({'sentence': matching_sentences})
    new_dataset.index.name = ''  # Remove the name from the index
    new_dataset.reset_index(inplace=True)  # Reset index to have the required format

    # Save to a CSV file
    new_dataset.to_csv('new_alb_the_reviewed_unfiltered.csv', index=False)
    print("Created dataset with 5000 sentences of the same token length.")