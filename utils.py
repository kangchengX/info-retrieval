from typing import Iterable, Tuple, Dict
import numpy as np
import pandas as pd
import nltk, re
import data
from nltk.stem import WordNetLemmatizer


Tag_To_Pos = {'J': 'a', 'V': 'v', 'R': 'r'}


def count(data: Iterable, frequencies_in_dicts: bool | None = True):
    '''Count each term in the data
    
    Args:
        data: data to count

    Returns:
        result: if frequencies_in_dicts : a dict with terms as keys and frequencies as values,\
            else: a dict with terms as keys and {'tf': frequency} as values
    '''

    result = {}

    if frequencies_in_dicts:
        for term in data:
            result[term]['tf'] = result.setdefault(term, {'tf':0})['tf'] + 1
    else:
        for term in data:
            result[term] = result.get(term,0) + 1

    return result


def calculate_normalized_frequency_zipfian(
        fre : dict, 
        s : int | None = 1,
        return_difference : bool | None = True
) -> Tuple[np.ndarray, np.ndarray, float] | Tuple[np.ndarray, np.ndarray]:
    '''Calculate the normalized frequencies and the zipfian value as well as their difference

    Args:
        fre: a dict representing the (unnormalized) frequencies of the terms 
        s: parameter of zipfian distribution
        return_difference : If true, return the difference between the normalized frequencies and the zipfian value,\
            defined as the mean of the l2 norm
    
    Returns:
        fre_norm: normalized frequencies with descending order
        zipfian: zipfian value with descending order
        diff: difference between the two distributions if return_difference is True
    '''

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


def generate_tokens(
        text: str, 
        pattern: re.Pattern, 
        lemmatizer: WordNetLemmatizer | None = None, 
        stopwords: Iterable | None = None, 
        vocabulary: Iterable | None = None
):
    '''Process the input string into list of lemmatized tokens

    Args: 
        text: the string to process
        pattern: the Pattern in re. Default is re.compile(r'[^a-zA-Z\s]').
        lemmatizer: the lemmatizer in nltk.stem to lemmatize the tokens. If None, not lemmatize.\
            Default is None
        stopwords: words to remove. If None, keep all tokens.\
            Default is None
        vocabulary: words to keep. If none, keep all tokens.\
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


def extract_passage_embedding(doc_loader: data.DocLoader, word_embedding: Dict[str, np.ndarray]):
    '''Extract passage vector embedding for each passage
    
    Args:
        doc_loader: the loader containing the passages
        word_embedding: a dict mapping the word to the vector

    Returns:
        passage_embedding: DataFrame with columns ('pid', 'vector')
    '''
    inverted_indices = doc_loader.inverted_indices
    passages_length = doc_loader.passages_length
    zero_vector = np.zeros(len(word_embedding['like']), dtype=word_embedding['like'].dtype)
    passage_embedding = {}

    # iter each term
    for term, feature_collection_dict in inverted_indices.items():
        word_vector = word_embedding.get(term, zero_vector)
        for pid, feature_passage_dict in feature_collection_dict['pids_dict'].items():
            passage_embedding[pid] = passage_embedding.get(pid, zero_vector) + feature_passage_dict['tf'] * word_vector
    
    # average the vectors
    passage_embedding = {pid: sum_vector / passages_length[pid] 
                         for pid, sum_vector in passage_embedding.items()}
    # convert dict to DataFrame
    passage_embedding = pd.DataFrame(list(passage_embedding.items()), columns=['pid', 'vector'])
    
    return passage_embedding


def extract_query_embedding(query_loader: data.QueryLoader, word_embedding: Dict[str, np.ndarray]):
    '''Extract query vector embedding for each query
    
    Args:
        query_loader: the loader containing the queries
        word_embedding: a dict mapping the word to the vector

    Returns:
        query_embedding: DataFrame with columns ('qid', 'vector')
    '''
    zero_vector = np.zeros(len(word_embedding['like']), dtype=word_embedding['like'].dtype)
    queries = query_loader.queries
    query_embedding =  {
        qid: np.mean(
            [
                word_embedding.get(term, zero_vector) * feature_dict['tf'] 
                for term, feature_dict in term_dict.items()
            ]
            )
        for qid, term_dict in queries.items()
    }
    
    query_embedding = pd.DataFrame(list(query_embedding.items()), columns=['qid','vector'])

    return query_embedding


def extract_features(
        data_loader: data.DataLoader,
        word_embedding: Dict[str, np.ndarray],
        query_candidate: pd.DataFrame,
        filename: str | None = None
):
    """Extract feature representation for the query-passage pair. Use inverted indices in data_loader \
    to improve efficiency. 
    
    Args:
        data_loader: the data loader containing the passages and queris
        word_embedding: a dict mapping the word to the vector
        query_candidate: DataFrame with columns (qid, pid)
        filename: filename to save the features. If None, not save the features

    Returns:
        features: the 
    """
    passage_embedding = extract_passage_embedding(data_loader.doc_loader, word_embedding)
    query_embedding = extract_query_embedding(data_loader.query_loader, word_embedding)

    features = query_candidate.merge(passage_embedding, on='pid').merge(query_embedding, on='qid')

    if filename is not None:
        features.to_csv(filename)

    return features


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

    Return:
        mean_ap : mean of the average precision
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
