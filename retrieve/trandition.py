from typing import Dict, Tuple, List, Callable
import numpy as np
import pandas as pd
import utils
from functools import partial

def _retrieve_traditional_part(queries_feature:List[Tuple[str,Dict[str,int]]], 
                               query_candidate:dict, 
                               socre_funct:Callable, 
                               print_details:bool):
    '''retrive according to the query dict and the partial score function

    Args:
        queries_features: tf_queries: a list of tuple (qid,value) where \
            values are dict with with all term within the query as keys and frequency or tf-tdf as values
        query_candidate: a dict with pids as keys and lists of cadicate passages as values
        socre_funct: the score fuction with two args. First is the query dict and second is the pid
        print_details: True indicate printing if retrive is finish for each query

    Returns:
        retrieve_all: retrieval resultes for all queries
    '''

    retrieve_all = pd.DataFrame()
    for qid, query_feature in queries_feature:
        retrieve = {}
        for pid in query_candidate[qid]:
            retrieve[pid] = socre_funct(query_feature, pid)

        retrieve = sorted(retrieve.items(), key=lambda item: item[1], reverse=True)[0:100]
        retrieve = pd.DataFrame(retrieve)
        retrieve.insert(loc = 0, column= None, value = qid)
        retrieve_all = retrieve_all.append(retrieve, ignore_index=True)
        if print_details:
            print('finish query {}'.format(qid))

    return retrieve_all


def cal_score_tfidf(tfidf_query:Dict[str,int], 
                    pid:str, 
                    inverted_indices:Dict[str,Dict[str,int]], 
                    idf:Dict[str,int]) -> np.float64:
    '''calulate the score for TF-IDF vector space based retrieval model

    Args:
        tfidf_query: a dict with terms within the query as keys and tf-idf as values
        pid: pid of the passage to query
        inverted_indices: inverted_indices: a dict whose keys are terms,\
            values are dicts with pids as keys and frequencies as values
        idf: a dict with terms as keys and idf as values
    
    Returns:
        score:  the socre of the query-passage pair for TF-IDF vector space based retrieval model
    '''
    product = 0
    norm1 = 0
    norm2 = 0

    for term, tfidf_query_term in tfidf_query.items():
        # only calculate for the common term
        if pid not in inverted_indices[term]:
            continue
        tfidf_passage_term = inverted_indices[term][pid] * idf[term]
        product += tfidf_query_term * tfidf_passage_term
        norm1 += tfidf_query_term ** 2
        norm2 += tfidf_passage_term ** 2

    # query and passage do not have common term
    if norm1 == 0:
        return 0

    score = product / np.sqrt(norm1) / np.sqrt(norm2)

    return(score)


def retrieve_vector_space(inverted_indices:Dict[str,Dict[str,int]],
                          tf_queries:List[Tuple[str,Dict[str,int]]],
                          query_candidate:dict, 
                          num_passages: int,
                          filename:str, 
                          print_details = False) -> None:
    '''implement the TF-IDF vector space based retrieval model

        Args: 
            inverted_indices: inverted_indices: a dict whose keys are terms,\
                values are dicts with pids as keys and frequencies as values
            tf_queries: a list of tuple (qid,value) where \
                values are dict with with all term within the query as keys and frequency as values
            query_candidate: a dict with query as keys and the lists of the pids as values
            num_passages: number of the passages (size of the collection)
            filename: the path of the .csv file storing the results of the model
            print_details: True indicate printing if retrive is finish for each query

        Files Created:
            a .csv file with (qid,pid,score) as rows, ordered by order of query in tf_idf_query, score DESC. For each qid it just keep top 100 pid
    '''
    print('start : TF-IDF vector space based retrieval model')

    cal_score_tfidf_partial = partial(cal_score_tfidf,
                                      inverted_indices = inverted_indices,
                                      idf = idf)
    
    idf = {k:np.log10(num_passages/len(v)) for k,v in inverted_indices.items()}

    tfidf_queries = utils.cal_tfidf_query(tf_queries, idf)

    retrieve_all = _retrieve_traditional_part(tfidf_queries, query_candidate, cal_score_tfidf_partial,print_details)
    
    retrieve_all.to_csv(filename,index=False,header=False)
    print('finish : TF-IDF vector space based retrieval model. Length is {}'.format(len(retrieve_all.index)))


def cal_score_BM25(tf_query:Dict[str,int], 
                   pid:str,
                   inverted_indices:Dict[str,Dict[str,int]], 
                   df:dict,
                   passage_length:int, 
                   ave_length: float, 
                   num_passages:int, 
                   k1:float, k2:float, b:float) -> float:
    '''calculate the score for BM25

    Args:
        tf_query: a dict with all terms within the query as keys, frequency as values
        pid: id of the passage
        inverted_indices: a dict representing inverted indicex. keys are all terms,\
            values are dicts with pid as keys and frequencies as values
        df: a dict with all terms as keys, the number of the passage containing the term as values
        passage_length: the length of the passage
        ave_length: average length of the passages
        num_passages: number of the passages / size of the collection
        k1, k2, b: hyparameters of the score function for BM25 

    Returns:
        score: the BM25 socre of the query-passage pair
    '''
    K = k1 * ((1 - b) + b * passage_length / ave_length)

    score = 0
    for term, tf_query_term in tf_query.items():
        if pid not in inverted_indices[term]:
            continue
        score += np.log(
            ((0 + 0.5) / (0 - 0 + 0.5)) / ((df[term] - 0 + 0.5 ) / (num_passages - df[term] - 0 + 0 + 0.5)) \
        * ((k1 + 1) * inverted_indices[term][pid]) / (K + inverted_indices[term][pid]) \
        * ((k2 + 1) * tf_query_term) / (k2 + tf_query_term)
        )

    return(score)


def retrieve_BM25(inverted_indices:Dict[str,Dict[str,int]],
                  tf_queries:List[Tuple[str,Dict[str,int]]],
                  query_candidate:dict, 
                  num_passages: int,
                  filename:str, 
                  k1:float, k2:float, b:float, 
                  print_details = False):
    '''implement BM25

    Args:
        inverted_indices: a dict representing inverted indicex. keys are all terms in the vobavulary,\
            values are dicts with pid as keys and frequencies as values
        tf_queries: a list of tuple (qid,value) where \
            values are dict with with all term within the query as keys and frequency as values
        query_candidate: a dict with query as keys and the lists of the pids as values
        num_passages: number of the passages (size of the collection)
        filename: .csv file path to save the results 
        k1,k2,b: hyparameters of the score function for BM25 
        print_details: True indicate printing if retrive is finish for each query

    Files Created:
        a .csv file with (qid,pid,score) as rows, ordered by order of quary in tf_idf_quary, score DESC. For each qid it just keep top 100 pid
    '''

    print('start : BM25')
    df = {k:len(v) for k,v in inverted_indices.items()}
    passage_length = utils.cal_passage_length(inverted_indices)
    ave_length = np.average(list(passage_length.values()))

    cal_score_BM25_partial = partial(cal_score_BM25,
                                     inverted_indices = inverted_indices,
                                     df = df,
                                     passage_length = passage_length,
                                     ave_length = ave_length,
                                     num_passages = num_passages,
                                     k1=k1, k2=k2, b=b)
    
    retrieve_all = _retrieve_traditional_part(tf_queries, 
                                              query_candidate, 
                                              cal_score_BM25_partial, 
                                              print_details)
    
    print('finfish : BM25')
    
    retrieve_all.to_csv(filename,index=False,header=False)


def cal_score_discounting(tf_query:Dict[str,int],
                          pid:str, 
                          inverted_indices:Dict[str,Dict[str,int]],
                          passages_length:Dict[str,int],
                          v:int, 
                          eps:float) -> float:
    '''calculate the log score under discounting smoothing

    Args:
        tf_query: a dict with all terms within the query as keys, frequency as values
        pid: id of the passage
        inverted_indices: a dict representing inverted indicex. keys are all terms in the vobavulary,\
            values are dicts with pid as keys and frequencies as values
        passages_length: a dict with pids as keys and length of the passage as values
        v: number of unique words in the entire collection
        eps: parammter for the smoothing method.

    Returns:
        score: the log score
    '''

    score = 0
    d = passages_length[pid]

    for term, term_count in tf_query:
        score += term_count * np.log((eps + inverted_indices.get(term,{}).get(pid,0)) / (eps * v+d))

    return score


def cal_score_dirichlet(tf_query:Dict[str,int], 
                        pid:str,
                        inverted_indices:Dict[str,Dict[str,int]], 
                        passages_length:Dict[str,int],
                        tf_collection:dict, 
                        length_collection:int, 
                        mu:float) -> float:
    '''calculate the log score under dirichlet smoothing

    Args:
        tf_query: a dict with all terms within the query as keys, frequency as values
        pid: id of the passage
        inverted_indices: a dict representing inverted indicex. keys are all terms in the vobavulary,\
            values are dicts with pid as keys and frequencies as values
        passages_length: a dict with pids as keys and length of the passage as values
        tf_collection: a dict with terms in vocabulary as keys and counts as values
        length_collection: number of terms in the entire collection
        mu: parammter for the smoothing method.

    Returns:
        score: the log score
    '''

    score = 0
    d = passages_length[pid]
    lam = d / (d + mu)

    for term, term_count in tf_query:
        score += term_count * np.log(lam * inverted_indices[term].get(pid,0) / d 
                        + (1-lam) * tf_collection[term] / length_collection)

    return score


def retrieve_query_liklihood(inverted_indices:Dict[str,Dict[str,int]],
                             tf_queries:List[Tuple[str,Dict[str,int]]],
                             query_candidate:dict, 
                             smooth_type:str, 
                             pram:float, 
                             filename_out:str, 
                             print_details:bool | None = False):
    '''implement the query_liklihood retreival model
    
    Args:
        inverted_indices: a dict representing inverted indicex. keys are all terms in the vobavulary,\
                values are dicts with pid as keys and frequencies as values
        tf_queries: a list of tuple (qid,value) where \
            values are dict with with all term within the query as keys and frequency as values
        query_candidate: a dict with pids as keys and lists of cadicate passages as values
        smooth_type: type of the smoothing method
        pram: prameter for the smoothing fuction
        filename_out: the path of the .csv where the results are stored
        print_details: True indicate printing if retrive is finish for each query

    Files Created:
        a .csv file with qid,pid,score as rows
    '''
    print('start : query liklihood retrieve using ' + smooth_type)

    passages_length = utils.cal_passage_length(inverted_indices)

    # calculate the frequencies of terms in the entire collection
    tf_collection = {term: sum(value.values()) for term, value in inverted_indices.items()}
    v = len(tf_collection)
    length_collection = sum(tf_collection.values())

    if smooth_type == 'discount':
        funct = partial(cal_score_discounting,
                        inverted_indices=inverted_indices,
                        passages_length=passages_length,
                        v=v,
                        eps=pram)
    elif smooth_type == 'interp':
        funct = partial(cal_score_dirichlet,
                        inverted_indices=inverted_indices,
                        passages_length=passages_length,
                        tf_collection=tf_collection,
                        length_collection=length_collection,
                        mu=pram,)
    else:
        raise ValueError('Unknown smoothing type')
    
    retrieve_all = _retrieve_traditional_part(tf_queries, query_candidate, funct, print_details)
    
    print('finish : query liklihood retrieve using ' + smooth_type)
    
    retrieve_all.to_csv(filename_out,index=False,header=False)
    