import os
import random
import spacy
from nltk.corpus import wordnet as wn
import nltk

# Ensure NLTK WordNet data is downloaded
nltk.download('wordnet')
# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
# set random seed for reproducibility
random.seed(525)


# Wordnet examples inspired from nltk documentation https://www.nltk.org/howto/wordnet.html and medium article:
# https://medium.com/@alshargi.usa/boosting-text-analysis-with-wordnet-synonyms-and-hypernyms-52f22ac58cd8

# Helper function to fix formatting issues on rebuilt text
def format_text(document, replacement):
    out=[]
    for token, rep in zip(document, replacement):
       # if token.whitespace_:
        out.append(rep + token.whitespace_)
    return "".join(out)
        #else:
         #   out.append(rep)

# Function to get a random synonym for a given word
def get_synonym(word):
    syns = []
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                syns.append(lemma.name().replace('_', ' '))
    return random.choice(syns) if syns else word

# Replace words in the document with their synonyms based on a given probability
def synonym_replacement(document, replacement_prob):
    new_words = []

    for token in document:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and random.random() < replacement_prob:
            synonym = get_synonym(token.lemma_)
            if synonym:
                new_words.append(synonym)
                continue
        new_words.append(token.text)

    return format_text(document, new_words)

# Optional Noise Based Augmentation to include random types for words over 5 characters long given a probability
def random_typo(word):
    if len(word) <= 5:
        return word

    typo_type = random.choice(["swap", "delete"])

    # use list slicing to create the typo based on type chosen
    if typo_type == "swap" and len(word) >=2:
        i = random.randint(0, len(word) - 2)
        return word[:i] + word[i+1] + word[i] + word[i+2:]
    else:
        i = random.randint(0, len(word) - 1)
        return word[:i] + word[i+1:]

def inject_noise(document, noise_prob):
    new_words = []

    for token in document:
        if token.is_alpha and random.random() <= noise_prob:
            new_word = random_typo(token.text)
            new_words.append(new_word)
        else:
            new_words.append(token.text)
    return format_text(document, new_words)

def augment_text(text, synonym_prob, noise_prob = None):

    doc = nlp(text)

    # Apply synonym replacement
    augmented_text = synonym_replacement(doc, synonym_prob)

    # Apply noise injection if noise_prob is provided
    if noise_prob is not None and noise_prob > 0:
        doc_with_synonyms = nlp(augmented_text)
        augmented_text = inject_noise(doc_with_synonyms, noise_prob)

    return augmented_text

def augment_dataset(lex_prob, noise_prob):
    input_folder = "dataset/test/"
    output_folder = "dataset/augmented_test/"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
                original = f.read()

            augmented = augment_text(original, lex_prob, noise_prob)

            outname = filename.replace(".txt", "_aug.txt")
            with open(os.path.join(output_folder, outname), "w", encoding="utf-8") as out:
                out.write(augmented)

            print(f"Augmented: {filename} → {outname}")

    print("\nAugmentation complete!")


def main():

    print("------ Module 5: Text Data Augmentation ------")
    print("\nThis program augments text data by replacing words with their synonyms and optionally injecting noise.\n")
    print("Noise injection introduces random typos in words longer than 5 characters.\n")
    syn_probabilty = int(input("Enter the probability percentage of synonym replacement for eligible words: %="))
    syn_probabilty /= 100.0
    use_noise = input("Do you want to include noise injection? (y/n): ")
    # Determine noise probability if user opts in
    if use_noise.lower() == 'y':
        noise_probabilty = int(input("Enter the probability percentage of noise injection for words longer than 5 characters: %="))
        noise_probabilty /= 100.0
    else:
        print("Proceeding without noise injection.")
        noise_probabilty = 0.0

    print(f"\nRunning augmentation with:")
    print(f"- Synonym replacement augmentation probability: {syn_probabilty}")
    if noise_probabilty == 0.0:
        print("- Noise augmentation: DISABLED")
    else:
        print(f"- Noise augmentation probability:   {noise_probabilty}\n")

    augment_dataset(syn_probabilty, noise_probabilty)

if __name__ == "__main__":
    main()


