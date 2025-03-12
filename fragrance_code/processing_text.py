import nltk
from nltk.corpus import wordnet
nltk.download("wordnet")
nltk.download("omw-1.4")
import pandas as pd
import re

def prepare_for_tfidf(text):
    """
    Function to clean and prepare fragrance notes for TF-IDF while keeping commas.
    
    Parameters:
        text (str): The input text containing fragrance notes.
        
    Returns:
        str: The cleaned and formatted text.
    """
    # Check if the input is NaN (missing value) and return an empty string if so
    if pd.isna(text):
        return ""
    
    # Remove special characters (™, ®, etc.), but KEEP commas for phrase separation
    text = re.sub(r"[^\w\s,]", "", text)  
    
    # Convert text to lowercase for consistency
    text = text.lower()

    # Replace hyphens with spaces to improve word recognition in TF-IDF
    text = text.replace("-", " ")

    # Remove extra spaces around commas (ensures "word, word" format)
    text = re.sub(r"\s*,\s*", ", ", text)

    # Remove any leading or trailing whitespace
    return text.strip()


def expand_query(query):
    
    """
    Expands the given query by adding synonyms from WordNet.

    This function takes a query string, splits it into individual words,
    retrieves synonyms for each word using WordNet, and adds them to the query.
    The expanded query is returned as a space-separated string.

    Args:
        query (str): The input query containing one or more words.

    Returns:
        str: The expanded query including the original words and their synonyms.
    
    """
    words = query.split()
    expanded_words = set(words)

    for word in words:
        synonyms = wordnet.synsets(word)
        for syn in synonyms:
            for lemma in syn.lemmas():
                expanded_words.add(lemma.name().replace("_", " "))

    return " ".join(expanded_words)