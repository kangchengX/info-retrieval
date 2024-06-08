import re
from typing import Dict, Tuple, List, Iterable
import numpy as np
import pandas as pd
import nltk
from constants import Tag_To_Pos
import matplotlib.pyplot as plt
from functools import partial

def count(data:Iterable):
    '''count each term in the data
    
    Args:
        data: data to count

    Returns:
        result: a dict with terms as keys and frequencies as values
    '''

    result = {}
    for term in data:
        result[term] = result.get(term,0) + 1

    return result


def generate_tokens(line:str, 
                    pattern:re.Pattern, 
                    lemmatizer = None, 
                    stopwords:Iterable | None=None, 
                    vocabulary:Iterable | None=None):
    '''process the input string into list of lemmatized tokens

    Args: 
        line: the string to process
        pattern: the Pattern in re
        lemmatizer: the lemmatizer in nltk.stem to lemmatize the tokens, None indicates not lemmatizing
        stopwords: words to remove. None indicates not removing any word.
        vocabulary: If not none, all the output tokens are in the vocabulary.

    Returns:
        tokens: a list of tokens
    '''

    tokens = pattern.sub(' ',line).lower().split()
    if stopwords is not None:
        tokens = [token for token in tokens if token not in stopwords]

    if lemmatizer is not None:
        tokens_tags = nltk.pos_tag(tokens)
        tokens = []
        pos = Tag_To_Pos.get(tag[0], 'n')
        for token, tag in tokens_tags:
            pos = Tag_To_Pos.get(tag[0], 'n')
            token = lemmatizer.lemmatize(token,pos)
            tokens.append(token)

    if vocabulary is not None:
        tokens = [token for token in tokens if token in vocabulary]

    return tokens


def extract_frequencies(filename:str, 
                        pattern:re.Pattern, 
                        lemmatizer = None, 
                        stopwords:Iterable | None=None, 
                        vocabulary:Iterable | None=None) -> dict:
    '''extract frequencies of terms (1-grams) from .txt file

    Args:
        filename: path of the .txt file
        remove: bool, True indicates removing stop words
    Returns:
        frequencies: a dict with tokens (terms) as keys and frequencies as values
    '''

    with open(filename,'r',encoding='utf-8') as f:
        content = f.read()

    tokens = generate_tokens(content, 
                             pattern=pattern, 
                             lemmatizer=lemmatizer, 
                             stopwords=stopwords,
                             vocabulary=vocabulary)

    frequencies = {}

    # count the frequencies
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1

    return frequencies


def norm_fre_zipfian(fre:dict, s = 1):
    '''get the normalized frequencies and the zipfian value as well as their difference

    Args:
        fre: a dict representing the (unnormalized) frequencies of the terms 
        s: parameter of zipfian distribution

    Returns:
        fre_norm: normalized frequencies with descending order
        zipfian: zipfian value with descending order
        diff: difference between the two distributions
    '''

    fre_sorted = sorted(fre.items(),key = lambda item:item[1],reverse= True)
    fre_norm = np.array([v for _,v in fre_sorted])
    fre_norm = fre_norm / np.sum(fre_norm)

    n = len(fre_norm)
    zipfian = np.arange(1,n+1,dtype='float') ** (-s) / np.sum(np.arange(1,n+1,dtype='float') ** (-s))

    diff = np.average((fre_norm -zipfian)**2)
    
    return fre_norm, zipfian, diff


def plot(fre_norm, zipfian, filename:str | None = None, show = True):
    '''plot the fre_norm and zipfian both in normal (linear) and log-log scales

    Args:
        fre_norm: a vector of normalized frequencies with descending order
        zipfian: a vector of zipfian values with descending order
        filename: None or the filename to save the plot. None indicates not saving the plot
        show: True indicates show the plot

    Files Created:
        a file with filename, showing the distrubution and zipfian comparsion both in log-log and normal scale
    '''

    x = np.arange(1,len(fre_norm)+1)

    _, axes = plt.subplots(1,2,figsize = (14, 6))

    # normal scale
    axes[0].plot(x,fre_norm,'b-',markersize=5,markerfacecolor='none', label = 'empirical')
    axes[0].plot(x,zipfian,'r--',markersize=5,markerfacecolor='none', label = "Zipf's law")
    axes[0].set_title('distribution comparison - normal')

    # log-log scale
    axes[1].loglog(x,fre_norm,'b-',markersize=5,markerfacecolor='none', label = 'empirical')
    axes[1].loglog(x,zipfian,'r--',markersize=5,markerfacecolor='none', label = "Zipf's law")
    axes[1].set_title('distribution comparison - log-log')

    for ax in axes:
        ax.set_xlabel(r'term frequency rank $k$')
        ax.set_ylabel(r'normalized frequency $f$')
        ax.legend()

    plt.savefig(filename)
    if show:
        plt.show()


def inverted_index(search_data:pd.DataFrame, 
                   pattern:re.Pattern, 
                   lemmatizer = None, 
                   stopwords:Iterable | None = None, 
                   vocabulary:Iterable | None = None):
    '''generate inverted indices for terms

    Args: 
        search_data: data to search, with row (pid,qid,query,passage)
        pattern: the Pattern in re
        lemmatizer: the lemmatizer in nltk.stem to lemmatize the tokens, None indicates not lemmatizing
        stopwords: words to remove. None indicates not removing any word.
        vocabulary: If not none, all the output tokens are in the vocabulary.
        
    Returns:
        inverted_indices: a dict whose keys are terms,\
            values are dicts with pids as keys and frequencies as values
    '''

    print('start : calculate inverted indices')

    # generate Series with pid as index and passage as values
    pid_passage = search_data.drop_duplicates('pid').set_index('pid')['passage']

    generate_tokens_partial = partial(generate_tokens, 
                                      pattern=pattern, 
                                      lemmatizer=lemmatizer, 
                                      stopwords=stopwords, 
                                      vocabulary=vocabulary)

    pid_tokens = pid_passage.apply(generate_tokens_partial)

    inverted_indices = {}

    for pid, tokens in pid_tokens.items():
        for token in tokens:
            pid_dict = inverted_indices.setdefault(token, {})
            pid_dict[pid] = pid_dict.get(pid,0) + 1


    print('finish : calculate inverted indices')

    return inverted_indices


def cal_tf_idf_passage(inverted_indices:Dict[str,Dict[str,int]], idf:Dict[str,float]) -> Dict[str,Dict[str,float]]:
    '''calculate the tf-idf for each term within the passage for each passage

    Args:
        inverted_indices: a dict whose keys are all terms in the vobavulary,\
            values are dicts with pids as keys and frequencies as values
        idf: a dict with terms as keys and idf as values

    Returns:
        tf_idf: a dict whose keys are pids,\
            values are dicts with all terms within the passage as keys and tf-idf as values
    '''

    print('start : calculate tf-idf for passages')
    tf_idf = {}
    for term, fre4eachpas in inverted_indices.items():
        idf_term = idf[term]
        for pid,fre in fre4eachpas.items():
            pid_dict = tf_idf.setdefault(pid,{})
            pid_dict[term] = fre * idf_term

    print('finish : calculate tf-idf for passages')
    return tf_idf


def cal_tf_query(query_df:pd.DataFrame,
                 pattern:re.Pattern, 
                 lemmatizer = None, 
                 stopwords:Iterable | None = None, 
                 vocabulary:Iterable | None = None) -> List[Tuple[str,Dict[str,int]]]:
    '''calculate the frequency for each term within one query
    
    Args:
        query_df: the DataFrame of query, with (qid,query) as rows
        pattern: the Pattern in re
        lemmatizer: the lemmatizer in nltk.stem to lemmatize the tokens, None indicates not lemmatizing
        stopwords: words to remove. None indicates not removing any word.
        vocabulary: If not none, all the output tokens are in the vocabulary.

    Returns:
        tf: a list of tuple (qid,value) where values are keys with all term within the query as keys and frequency as values
    '''

    print('start : calculate tf for queries')
    generate_tokens_partial = partial(generate_tokens, 
                                      pattern=pattern, 
                                      lemmatizer=lemmatizer, 
                                      stopwords=stopwords, 
                                      vocabulary=vocabulary)
    
    query_se = query_df.set_index('qid')['query']

    tf = query_se.apply(generate_tokens_partial).apply(count)
    tf = list(tf.items())

    print('finish : calculate tf for queries')

    return tf


def cal_tfidf_query(tf:List[Tuple[str,Dict[str,int]]], idf:Dict[str,float]) -> List[Tuple[str,Dict[str,float]]]:
    '''calculate the tf-idf for each term within the query & preprocessed passages for each query

    Args:
        tf: a list of tuple (qid,value) where values are dicts with all term within the query as keys and frequency as values
        idf: a dict with terms in preprocessed passages as keys and idf as values

    Returns:
        tf_idf: a list of tuple (qid,value) where values are keys with all terms of the query & vocabulary as keys and tf-idf as values
    '''

    tf_idf = []

    print('start : calculate tf-idf for queries')

    for qid,tf_query in tf:
        tf_idf_dict = {}
        for term, tf_term in tf_query.items():
            tf_idf_dict[term] = tf_term * idf[term]

        tf_idf.append((qid,tf_idf_dict))

    print('finish : calculate tf-idf for queries')

    return tf_idf


def cal_passage_length(inverted_indices:Dict[str,Dict[str,int]]) -> Dict[str,int]:
    '''calculate length, i.e. number of items (or tokens), for each passage)
    
    Args: 
        inverted_indices: a dict representing inverted indicex. keys are all terms in the collection,\
            values are dicts with pid as keys and frequencies as values
    
    Returns:
        passages_length: a dict with pids as keys and length of the passage as values
    '''

    passages_length = {}
    for counts in inverted_indices.values():
        for pid, count in counts.items():
            passages_length[pid] = passages_length.get(pid,0) + count

    return passages_length

def cal_ndcg(relevance:pd.Series):
    '''calculate ndcg for the retrival results for one query

    Args:
        relevance: retreived results, (indices,relevancy)

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


def cal_ap(relevance:pd.Series):
    '''calculate the average precision of the retrived results
    
    Args:
        relevance: retreived results, (indices,relevancy)

    Returns:
        ap: average precision
    '''

    relevancies = relevance.values
    num_relevant = np.cumsum(relevancies)
    num_all = np.cumsum(np.ones(len(num_relevant)))
    precisions = num_relevant / num_all

    ap = np.mean(precisions * relevancies)

    return ap


def cal_metrics(results:pd.DataFrame, qid_pid_relevance:pd.DataFrame | None=None):
    '''calculate average precision and ndcg for the retrived results
    
    Args: 
        results: retrival resutls, (qid,pid,score,relevancy) or (qid,pid,score)
        qid_pid_relevances: true relevance, (qid,pid,relevancy)

    Return:
        precision_avg : avg precision, or map
        ndcg: ndcg value
    '''

    if qid_pid_relevance is not None:
        results = pd.merge(results,qid_pid_relevance,on=['qid','pid'])

    results_group = results.groupby('qid')

    aps = results_group['relevancy'].apply(cal_ap)
    ndcgs = results_group['relevancy'].apply(cal_ndcg)
    
    map = aps.mean()
    ndcg = ndcgs.mean()

    return map, ndcg


# def cal_score_tfidf(tfidf_query:Dict[str,int], 
#                     pid:str, 
#                     inverted_indices:Dict[str,Dict[str,int]], 
#                     idf:Dict[str,int]) -> np.float64:
#     '''calulate the score for TF-IDF vector space based retrieval model

#     Args:
#         tfidf_query: a dict with terms within the query as keys and tf-idf as values
#         pid: pid of the passage to query
#         inverted_indices: inverted_indices: a dict whose keys are terms,\
#             values are dicts with pids as keys and frequencies as values
#         idf: a dict with terms as keys and idf as values
    
#     Returns:
#         score:  the socre of the query-passage pair for TF-IDF vector space based retrieval model
#     '''
#     product = 0
#     norm1 = 0
#     norm2 = 0

#     for term, tfidf_query_term in tfidf_query.items():
#         # only calculate for the common term
#         if pid not in inverted_indices[term]:
#             continue
#         tfidf_passage_term = inverted_indices[term][pid] * idf[term]
#         product += tfidf_query_term * tfidf_passage_term
#         norm1 += tfidf_query_term ** 2
#         norm2 += tfidf_passage_term ** 2

#     # query and passage do not have common term
#     if norm1 == 0:
#         return 0

#     score = product / np.sqrt(norm1) / np.sqrt(norm2)

#     return(score)


# def _retrieve_traditional_part(queries_feature:List[Tuple[str,Dict[str,int]]], query_candidate, socre_funct, print_details):

#     retrieve_all = pd.DataFrame()
#     for qid, query_feature in queries_feature:
#         retrieve = {}
#         for pid in query_candidate[qid]:
#             retrieve[pid] = socre_funct(query_feature, pid)

#         retrieve = sorted(retrieve.items(), key=lambda item: item[1], reverse=True)[0:100]
#         retrieve = pd.DataFrame(retrieve)
#         retrieve.insert(loc = 0, column= None, value = qid)
#         retrieve_all = retrieve_all.append(retrieve, ignore_index=True)
#         if print_details:
#             print('finish query {}'.format(qid))

#     return retrieve_all

# def retrieve_vector_space(inverted_indices:Dict[str,Dict[str,int]],
#                           tf_queries:List[Tuple[str,Dict[str,int]]],
#                           query_candidate:dict, 
#                           num_passages: int,
#                           filename:str, 
#                           print_details = False) -> None:
#     '''implement the TF-IDF vector space based retrieval model

#         Args: 
#             inverted_indices: inverted_indices: a dict whose keys are terms,\
#                 values are dicts with pids as keys and frequencies as values
#             tf_queries: a list of tuple (qid,value) where \
#                 values are dict with with all term within the query as keys and frequency as values
#             query_candidate: a dict with query as keys and the lists of the pids as values
#             num_passages: number of the passages (size of the collection)
#             filename: the path of the .csv file storing the results of the model
#             print_details: True indicate printing if retrive is finish for each query

#         Files Created:
#             a .csv file with (qid,pid,score) as rows, ordered by order of query in tf_idf_query, score DESC. For each qid it just keep top 100 pid
#     '''
#     print('start : TF-IDF vector space based retrieval model')

#     cal_score_tfidf_partial = partial(cal_score_tfidf,
#                                       inverted_indices = inverted_indices,
#                                       idf = idf)
    
#     idf = {k:np.log10(num_passages/len(v)) for k,v in inverted_indices.items()}

#     tfidf_queries = cal_tfidf_query(tf_queries, idf)

#     retrieve_all = _retrieve_traditional_part(tfidf_queries, query_candidate, cal_score_tfidf_partial,print_details)
    
#     retrieve_all.to_csv(filename,index=False,header=False)
#     print('finish : TF-IDF vector space based retrieval model. Length is {}'.format(len(retrieve_all.index)))


# def cal_score_BM25(tf_query:Dict[str,int], 
#                    pid:str,
#                    inverted_indices:Dict[str,Dict[str,int]], 
#                    df:dict,
#                    passage_length:int, 
#                    ave_length: float, 
#                    num_passages:int, 
#                    k1:float, k2:float, b:float) -> float:
#     '''calculate the score for BM25

#     Args:
#         tf_query: a dict with all terms within the query as keys, frequency as values
#         pid: id of the passage
#         inverted_indices: a dict representing inverted indicex. keys are all terms,\
#             values are dicts with pid as keys and frequencies as values
#         df: a dict with all terms as keys, the number of the passage containing the term as values
#         passage_length: the length of the passage
#         ave_length: average length of the passages
#         num_passages: number of the passages / size of the collection
#         k1, k2, b: hyparameters of the score function for BM25 

#     Returns:
#         score: the BM25 socre of the query-passage pair
#     '''
#     K = k1 * ((1 - b) + b * passage_length / ave_length)

#     score = 0
#     for term, tf_query_term in tf_query.items():
#         if pid not in inverted_indices[term]:
#             continue
#         score += np.log(
#             ((0 + 0.5) / (0 - 0 + 0.5)) / ((df[term] - 0 + 0.5 ) / (num_passages - df[term] - 0 + 0 + 0.5)) \
#         * ((k1 + 1) * inverted_indices[term][pid]) / (K + inverted_indices[term][pid]) \
#         * ((k2 + 1) * tf_query_term) / (k2 + tf_query_term)
#         )

#     return(score)


# def retrieve_BM25(inverted_indices:Dict[str,Dict[str,int]],
#                   tf_queries:List[Tuple[str,Dict[str,int]]],
#                   query_candidate:dict, 
#                   num_passages: int,
#                   filename:str, 
#                   k1:float, k2:float, b:float, 
#                   print_detail = False):
#     '''implement BM25

#     Args:
#         inverted_indices: a dict representing inverted indicex. keys are all terms in the vobavulary,\
#             values are dicts with pid as keys and frequencies as values
#         tf_queries: a list of tuple (qid,value) where \
#             values are dict with with all term within the query as keys and frequency as values
#         query_candidate: a dict with query as keys and the lists of the pids as values
#         num_passages: number of the passages (size of the collection)
#         filename: .csv file path to save the results 
#         k1,k2,b: hyparameters of the score function for BM25 
#         print_detail: True indicate printing if retrive is finish for each query

#     Files Created:
#         a .csv file with (qid,pid,score) as rows, ordered by order of quary in tf_idf_quary, score DESC. For each qid it just keep top 100 pid
#     '''

#     print('start : BM25')
#     df = {k:len(v) for k,v in inverted_indices.items()}
#     passage_length = cal_length(inverted_indices)
#     ave_length = np.average(list(passage_length.values()))

#     cal_score_BM25_partial = partial(cal_score_BM25,
#                                      inverted_indices = inverted_indices,
#                                      df = df,
#                                      passage_length = passage_length,
#                                      ave_length = ave_length,
#                                      num_passages = num_passages,
#                                      k1=k1, k2=k2, b=b)
    
#     retrieve_all = _retrieve_traditional_part(tf_queries, query_candidate, cal_score_BM25_partial, print_detail)
    
#     print('finfish : BM25')
    
#     retrieve_all.to_csv(filename,index=False,header=False)


# ####### query likely hood #######

# def cal_score_discounting(pid:str, 
#                           tf_query:Dict[str,int], 
#                           inverted_indices:Dict[str,Dict[str,int]],
#                           passages_length:Dict[str,int],
#                           v:int, 
#                           eps) -> float:
#     '''calculate the log score under discounting smoothing

#     Args:
#         pid: id of the passage
#         tf_query: a dict with all terms within the query as keys, frequency as values
#         inverted_indices: a dict representing inverted indicex. keys are all terms in the vobavulary,\
#             values are dicts with pid as keys and frequencies as values
#         passages_length: a dict with pids as keys and length of the passage as values
#         v: number of unique words in the entire collection
#         eps: parammter for the smoothing method.

#     Returns:
#         score: the log score
#     '''

#     score = 0
#     d = passages_length[pid]

#     for term, term_count in tf_query:
#         score += term_count * np.log((eps + inverted_indices.get(term,{}).get(pid,0)) / (eps * v+d))

#     return score


# def cal_score_dirichlet(pid:str,
#                         tf_query:Dict[str,int], 
#                         inverted_indices:Dict[str,Dict[str,int]], 
#                         passages_length:Dict[str,int],
#                         tf_collection:dict, 
#                         mu, 
#                         length_collection) -> float:
#     '''calculate the log score under dirichlet smoothing

#     Args:
#         pid: id of the passage
#         tf_query: a dict with all terms within the query as keys, frequency as values
#         inverted_indices: a dict representing inverted indicex. keys are all terms in the vobavulary,\
#             values are dicts with pid as keys and frequencies as values
#         passages_length: a dict with pids as keys and length of the passage as values
#         tf_collection: a dict with terms in vocabulary as keys and counts as values
#         mu: parammter for the smoothing method.
#         length_collection: number of terms in the entire collection

#     Returns:
#         score: the log score
#     '''

#     score = 0
#     d = passages_length[pid]
#     lam = d / (d + mu)

#     for term, term_count in tf_query:
#         score += term_count * np.log(lam * inverted_indices[term].get(pid,0) / d 
#                         + (1-lam) * tf_collection[term] / length_collection)

#     return score


# def retrieve_query_liklihood(inverted_indices:Dict[str,Dict[str,int]],
#                              tf_queries:List[Tuple[str,Dict[str,int]]],
#                              query_candidate:dict, 
#                              smooth_type:str, 
#                              pram, 
#                              filename_out:str, 
#                              print_details = False):
#     '''implement the query_liklihood retreival model
    
#     Args:
#         queries: a list of tuple(qid,tokens) where tokens are listed of the terms (not unique) within the query
#         query_candidate: a dict with pids as keys and lists of cadicate passages as values
#         smooth_type: type of the smoothing method
#         pram: prameter for the smoothing fuction
#         filename_out: the path of the .csv where the results are stored

#     Output:
#         (not return): a .csv file with qid,pid,score as rows
#     '''
#     print('start : query liklihood retrieve using ' + smooth_type)

#     passages_length = cal_passage_length(inverted_indices)

#     # calculate the frequencies of terms in the entire collection
#     tf_collection = {term: sum(value.values()) for term, value in inverted_indices.items()}
#     v = len(tf_collection)
#     length_collection = sum(tf_collection.values())

#     if smooth_type == 'discount':
#         funct = partial(cal_score_discounting,
#                         inverted_indices=inverted_indices,
#                         passages_length=passages_length,
#                         v=v,
#                         eps=pram)
#     elif smooth_type == 'interp':
#         funct = partial(cal_score_dirichlet,
#                         inverted_indices=inverted_indices,
#                         passages_length=passages_length,
#                         tf_collection=tf_collection,
#                         mu=pram,
#                         length_collection=length_collection)
#     else:
#         raise ValueError('Unknown smoothing type')
    
#     retrieve_all = _retrieve_traditional_part(tf_queries, query_candidate, funct, print_details)
    
#     print('finish : query liklihood retrieve using ' + smooth_type)
    
#     retrieve_all.to_csv(filename_out,index=False,header=False)
