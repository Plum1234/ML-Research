import nltk
from transformers import GPT2Tokenizer, GPT2Model
import csv
from nltk.corpus import wordnet
nltk.download('wordnet')
# Load the GPT-2 tokenizer and model
# Comes with the Hugging Face Model from their website
nltk.download('wordnet')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Function to find a synonym that is a single token
def find_single_token_synonym(word):
    synonyms = wordnet.synsets(word)
    no_word = "synonym"
    for synonym in synonyms:
        lemma = synonym.lemmas()[0].name()
        if len(tokenizer.tokenize(lemma)) == 1:
            return lemma
    return no_word
    # if there's no synonym, just put the word synonym for now
    # (can fix later with better dictionary implementation)

# Replace multi-token last words
def replace_last_word(sentence):
    sentence_tokens = tokenizer.tokenize(sentence)
    words = sentence.strip().split()
    # print("Sentence tokens: ", sentence_tokens)
    last_word = words[-1] 
    if len(sentence_tokens) > 1:
        last_token = sentence_tokens[-1]
        decoded_last_token = tokenizer.decode(tokenizer.convert_tokens_to_ids([last_token]))
        if last_word.strip() != decoded_last_token.strip():
            synonym = find_single_token_synonym(last_word)
            words[-1] = synonym
            replaced_sentence = ' '.join(words)
            print ("two token:", decoded_last_token, "word:", last_word)
            print(decoded_last_token)
            print(last_word)
            print(synonym)
            return replaced_sentence
        # else:
        #     print ("one token:", decoded_last_token)
    return sentence

# File paths
input_file = 'aob.csv'
output_file = 'new_file2.csv'

# Process the CSV file
with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        row_number = row[fieldnames[0]]  # Assuming the first column is the row number
        original_sentence = row['original']
        contrastive_sentence = row['contrastive']
        processed_original = replace_last_word(original_sentence)
        processed_contrastive = replace_last_word(contrastive_sentence)
        writer.writerow({fieldnames[0]: row_number, 'original': processed_original, 'contrastive': processed_contrastive})
        #split the two columns, run the tokenizer on both and get new sentences
        #write the sentence in the file, repeat.