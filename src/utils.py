import numpy as np
from preprocessing import preprotext

def build_vocab(documents):
    """Building list of words from train data for model. The function could be used as:
    vocab = build_vocab(cleaned_texts)
    len(vocab), vocab

    Args:
        documents: LIST of 

    Returns:
        list: vocabulary
    """
    vocab = []
    for document in documents:
        tokens = preprotext(document) # resulting tokens of each documents
        for token in tokens:
            vocab.append(token)
    
    vocab = list(set(vocab))
    vocab = {word: index for index, word in enumerate(vocab)}
    return vocab

def build_bow(cleaned_texts, vocab):
    """Build BoW to a token of clened texts
    doc_bow = [] # bow for each documents
    for docs in cleaned_texts:
        docs = build_bow(docs)
        doc_bow.append(docs)
    """
    bow = np.zeros((len(vocab)))
    for token in cleaned_texts:
        if token in vocab:
            bow[vocab[token]] += 1
    return bow