import numpy as np
import pandas as pd
import nltk, re
from typing import Iterable, Tuple, Dict
from functools import lru_cache
from nltk.stem import WordNetLemmatizer


Tag_To_Pos = {'J': 'a', 'V': 'v', 'R': 'r'}


def calculate_normalized_frequency_zipfian(
        fre : dict, 
        s : int | None = 1,
        return_difference : bool | None = True
) -> Tuple[np.ndarray, np.ndarray, float] | Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the normalized frequencies and the zipfian value as well as their difference.

    Args:
        fre (dict): A dictionary representing the (unnormalized) frequencies of the terms.
        s (float): Parameter of the zipfian distribution.
        return_difference (bool): If True, return the difference between the normalized frequencies and 
            the zipfian value, defined as the mean of the L2 norm. Defaults to False.

    Returns:
        fre_norm (list): Normalized frequencies in descending order.
        zipfian (list): Zipfian values in descending order.
        diff (float, optional): Difference between the two distributions if return_difference is True.
    """

    fre_sorted = sorted(fre.items(),key = lambda item:item[1],reverse= True)
    fre_norm = np.array([v for _,v in fre_sorted])
    fre_norm = fre_norm / np.sum(fre_norm)

    n = len(fre_norm)
    zipfian = np.arange(1, n+1, dtype=float) ** (-s) / np.sum(np.arange(1,n+1,dtype=float) ** (-s))
    
    if return_difference : 
        diff = np.average((fre_norm -zipfian)**2)
        
        return fre_norm, zipfian, diff

    else:
        return fre_norm, zipfian


class Lemmatizer:
    """
    This Python class initializes a lemmatizer with an option to use caching for improved performance.
    
    Args:
        use_cache (bool): The `use_cache` parameter in the `__init__` method is a boolean parameter that
            determines whether caching should be used for lemmatization. If `use_cache` is set to `True`, the
            lemmatization results will be cached using an LRU cache with a specified size. Defaults to True.
        cache_size (int): The `cache_size` parameter in the `__init__` method represents the maximum number of
            items that can be stored in the cache when caching is enabled (`use_cache=True`). This parameter
            determines the size of the cache and specifies how many recent function calls and their results will
            be stored for faster retrieval. Defaults to 10000.
    """
    def __init__(self, use_cache=True, cache_size=10000):
        self.use_cache = use_cache
        self.lemmatizer = WordNetLemmatizer()
        
        if self.use_cache:
            self._lemmatize = lru_cache(maxsize=cache_size)(self._lemmatize_uncached)
        else:
            self._lemmatize = self._lemmatize_uncached

    def _lemmatize_uncached(self, word:str, pos:str | None = 'n'):
        """
        The function `_lemmatize_uncached` takes a word and its part of speech (POS) as input and returns
        the lemmatized form of the word using a lemmatizer.
        
        Args:
            word (str): The `word` parameter in the `_lemmatize_uncached` method is a string that represents
                the word to be lemmatized.
            pos (str): The `pos` parameter in the `_lemmatize_uncached` method is used to specify the
                part of speech of the word being lemmatized. It is an optional parameter with a default value of
                'n', which stands for noun. This parameter allows you to specify the part of speech. Defaults to n
        
        Returns:
          The `_lemmatize_uncached` method is returning the lemmatized form of the input `word` with the
        specified part of speech `pos` using the lemmatizer.
        """
        return self.lemmatizer.lemmatize(word, pos)

    def lemmatize(self, word:str, pos:str | None = 'n'):
        return self._lemmatize(word, pos)
    
    def clear_cache(self):
        if self.use_cache:
            self._lemmatize.cache_clear()


def generate_tokens(
        text: str, 
        pattern: re.Pattern, 
        lemmatizer: Lemmatizer | None = None, 
        stopwords: Iterable | None = None, 
        vocabulary: Iterable | None = None
):
    '''Process the input string into list of lemmatized tokens

    Args: 
        text: the string to process
        pattern: the Pattern in re. Default is re.compile(r'[^a-zA-Z\\s]').
        lemmatizer: the lemmatizer to lemmatize the tokens. If None, not lemmatize.
            Default is None
        stopwords: words to remove. If None, keep all tokens.
            Default is None
        vocabulary: words to keep. If none, keep all tokens.
            Default is None

    Returns:
        tokens: a list of tokens
    '''

    tokens = pattern.sub(' ', text).lower().split()
    if stopwords is not None:
        tokens = [token for token in tokens if token not in stopwords]

    if lemmatizer is not None:
        tokens_tags = nltk.pos_tag(tokens)
        tokens = []
        for token, tag in tokens_tags:
            pos = Tag_To_Pos.get(tag[0], 'n')
            token = lemmatizer.lemmatize(token,pos)
            tokens.append(token)

    if vocabulary is not None:
        tokens = [token for token in tokens if token in vocabulary]

    return tokens


def cal_ndcg(relevance: pd.Series):
    '''Calculate ndcg for the retrival results for one query

    Args:
        relevance: the relevance scores of the retrieved results for this query

    Returns:
        ndcg: ndcg for the query
    '''
    dcg = 0
    for i,score in enumerate(relevance):
        dcg +=  (2**score -1) / np.log2(2+i)

    if dcg == 0:
        return 0

    relevance = relevance.sort_values(ascending=False)

    idcg = 0
    for i,score in enumerate(relevance):
        idcg +=  (2**score -1) / np.log2(2+i)

    ndcg = dcg / idcg

    return ndcg


def cal_average_precision(relevance: pd.Series):
    '''Calculate the average precision of the retrived results
    
    Args:
        relevance: the relevance scores of the retrieved results for this query

    Returns:
        ap: average precision
    '''

    relevancies = relevance.values
    num_relevant = np.cumsum(relevancies)
    num_all = np.cumsum(np.ones(len(num_relevant)))
    precisions = num_relevant / num_all

    ap = np.mean(precisions * relevancies)

    return ap


def cal_metrics(results: pd.DataFrame, qid_pid_relevance: pd.DataFrame | None = None):
    '''calculate average precision and ndcg for the retrived results
    
    Args: 
        results: retrival resutls, (qid,pid,score,relevancy) or (qid,pid,score)
        qid_pid_relevances: true relevance, (qid,pid,relevancy)

    Returns:
        mean_ap: mean of the average precision
        ndcg: ndcg value
    '''

    if qid_pid_relevance is not None:
        results = pd.merge(results,qid_pid_relevance,on=['qid','pid'])

    results_group = results.groupby('qid')

    aps = results_group['relevancy'].apply(cal_average_precision)
    ndcgs = results_group['relevancy'].apply(cal_ndcg)
    
    mean_ap = aps.mean()
    ndcg = ndcgs.mean()

    return mean_ap, ndcg
