import numpy as np
from src.preprocessing import preprotext

def build_vocab(documents):
    """Build vocabulary from given documents

    Args:
        documents (list): list of documents; could be df['text']

    Returns:
        dict: Vocabulary of words with index
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
    """Build Bag-of-Words to a token of clened texts

    Args:
        cleaned_texts (list): list of preprocessed (cleaned) documents (texts)
        vocab (dict): vocabulary generated from documents

    Returns:
        list: count of the same words existed in cleaned_texts and vocab
    """
    bow = np.zeros((len(vocab)))
    for token in cleaned_texts:
        if token in vocab:
            bow[vocab[token]] += 1
    return bow

# TF-IDF Algorithm

def calcTermFrequency(documents):
    termFreqList = []
    for document in documents:
        tf_doc = {} # term frequency for every documents
        for token in document:
            tf_doc[token] = tf_doc.get(token, 0) + 1
        termFreqList.append(tf_doc)
    return termFreqList

def calcDocFrequency(termFreqList):
    df = {}
    for tf_doc in termFreqList:
        for token in tf_doc.keys():
            df[token] = df.get(token, 0) + 1
    return df

def calcIDF(N, docFreq):
    idfList = {}
    for df in docFreq:
        idf_df = np.log10(N/docFreq.get(df))
        idfList.update({df: idf_df})
    return idfList

def calcTFIDF(termFreq, docFreq, idf):
    tf_idf = [] 
    for tf in termFreq:
        tf_idf_doc = {}
        for tokens in tf:
            tf_idf_doc[tokens] = docFreq[tokens] * idf[tokens]
        tf_idf.append(tf_idf_doc)
    return tf_idf