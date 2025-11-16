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

# Helper function to fix formatting issues on rebuilt text --  Modification of ChatGPT suggestion
def format_text(document, replacement):
    out=[]
    for token, rep in zip(document, replacement):
        out.append(rep + token.whitespace_)
    return "".join(out)


# Function to get a random synonym for a given word
def get_synonym(word):
    syns = []
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                syns.append(lemma.name().replace('_', ' '))
    return random.choice(syns) if syns else word


# Replace words in the document with their synonyms based on a given probability and part of speech (pos_list)
def synonym_replacement(document, pos_list, replacement_prob):
    new_words = []

    for token in document:
        if token.pos_ in pos_list and random.random() < replacement_prob:
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


# Inject noise into the document based on a given probability
def inject_noise(document, noise_prob):
    new_words = []

    for token in document:
        if token.is_alpha and random.random() <= noise_prob:
            new_word = random_typo(token.text)
            new_words.append(new_word)
        else:
            new_words.append(token.text)
    return format_text(document, new_words)


# Main augmentation function combining synonym replacement and optional noise injection
def augment_text(text, synonym_prob, noise_prob, pos_list):

    doc = nlp(text)

    # Apply synonym replacement
    augmented_text = synonym_replacement(doc, pos_list, synonym_prob)

    # Apply noise injection if noise_prob is provided
    if noise_prob is not None and noise_prob > 0:
        doc_with_synonyms = nlp(augmented_text)
        augmented_text = inject_noise(doc_with_synonyms, noise_prob)

    return augmented_text


# Function to process all text files in the input folder and save augmented versions
def augment_dataset(lex_prob, noise_prob, pos_list):
    input_folder = "dataset/original/"
    output_folder = "dataset/augmented/"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
                original = f.read()

            augmented = augment_text(original, lex_prob, noise_prob, pos_list)

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
    print("Which part of speech would you like to augment?")
    print("Options: NOUN, VERB, ADJ, ADV\n")
    # Dictionary of valid POS tags
    pos_dict = {
        '1': 'NOUN',
        '2': 'VERB',
        '3': 'ADJ',
        '4': 'ADV',
        '5': 'ALL'
    }
    pos_choices = []
    choice_completed = False
    # Loop until user is done selecting POS types
    while not choice_completed:
        print("1. NOUN\n2. VERB\n3. ADJ\n4. ADV\n5. All of the above")
        selection_choice = input("Enter your choice: ").strip()
        if selection_choice not in pos_dict:
            print("Invalid selection. Please try again.\n")
            continue
        selected_pos = pos_dict[selection_choice]
        # If user chooses ALL, override and finish immediately
        if selected_pos == 'ALL':
            pos_choices = ['NOUN', 'VERB', 'ADJ', 'ADV']
            choice_completed = True
            continue
        # Add the POS only if it's not already chosen
        if selected_pos not in pos_choices:
            pos_choices.append(selected_pos)
            print(f"Added: {selected_pos}")
        else:
            print(f"{selected_pos} already selected. Skipping.")

        # Ask if they want to choose more POS types
        quit_choice = input("Add another? (y/n): ").strip().lower()
        if quit_choice != 'y':
            choice_completed = True

    print("\nFinal POS selection:", pos_choices)

    # Ask user if they want to include noise injection
    use_noise = input("Do you want to include noise injection? (y/n): ")
    # Determine noise probability if user opts in
    if use_noise.lower() == 'y':
        noise_probability = int(input("Enter the probability percentage of noise injection for words longer than 5 characters: %="))
        noise_probability /= 100.0
    else:
        print("Proceeding without noise injection.")
        noise_probability = 0.0

    print(f"\nRunning augmentation with:")
    print(f"- Synonym replacement augmentation probability: {syn_probabilty}")
    if noise_probability == 0.0:
        print("- Noise augmentation: DISABLED")
    else:
        print(f"- Noise augmentation probability:   {noise_probability}\n")

    augment_dataset(syn_probabilty, noise_probability, pos_choices)


if __name__ == "__main__":
    main()
