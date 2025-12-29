import string
def preprotext(text): # a cool name for preprocessing-text
    """Cleaning text input by applying lowercase, removing punctuation, and tokenizing (text.split()). The function could be used as:

    cleaned_texts = texts.copy()
    for i in range(len(texts)):
        cleaned_texts[i] = preprotext(texts[i])

    Args:
        text: string to be cleaned.

    Returns:
        list: tokens of cleaned text.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation marks
    tokens = text.split() # tokenizing
    return tokens