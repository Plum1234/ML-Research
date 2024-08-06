import nltk
from nltk.corpus import wordnet

# Ensure WordNet is downloaded
nltk.download('wordnet')

def get_synonyms(word):
    synonyms = list()
    
    # Get synsets for the word
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    synonyms.pop(0)
    lol = synonyms[0]
    return lol

# Example usage
word = "toolbox"
synonyms = get_synonyms(word)
print("Synonyms for", word, ":", synonyms)