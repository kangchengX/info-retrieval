import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Iterable
import nltk
from nltk.corpus import stopwords


def to_tokens(line:str, pattern:re.Pattern, stop_words:Iterable[int]):
    '''conver line to tokens
    
    Args: 
        line: the str to process
        pattern: the re patttern
        stop_words: the stop words

    Returns:
        tokens: tokens
    '''

    tokens = pattern.sub(' ',line).lower().split()
    tokens = [token for token in tokens if token not in stop_words]

    return tokens


def invert_index(passages:pd.Series):
    '''generate inverted indices

    Args: 
        passages: (pid,list of tokens)
        
    Returns:
        inverted_indices: a dict whose keys are tokens,\
            values are dicts with pids as keys and frequencies as values
    '''

    print('start : calculate inverted indices')

    inverted_indices = {}

    for pid, tokens in passages.items():
        # calculate inverted indices
        for token in tokens:
            if token not in inverted_indices:
                inverted_indices[token] = {}
            if pid not in inverted_indices[token]:
                inverted_indices[token][pid] = 1
            else:
                inverted_indices[token][pid] += 1

    print('finish : calculate inverted indices')

    return inverted_indices


def cal_idf(inverted_indices:Dict[str,Dict[str,int]], num:int) -> Dict[str,int]:
    ''' calculate the idf value

    Input:
        inverted_indices: a dict whose keys are all tokens,\
            values are dicts with pids as keys and frequencies as values
        num: number of the passages (size the collection)

    Output:
        idf : a dict with terms as keys and idf as values
    '''

    idf = {k:np.log10(num/len(v)) for k,v in inverted_indices.items()}
    print('finish : calculate idf')

    return idf


def cal_length(inverted_indices:Dict[str,Dict[str,int]]) -> dict:
    '''calculate the length of each passage
    
    Input:
        inverted_indices: a dict representing inverted indices. keys are all tokens,\
            values are dicts with pid as keys and frequencies as values

    Output:
        passage_length: a dict with pid as keys and lengths as values 
    '''

    passage_length = {}
    for value in inverted_indices.values():
        for pid,count in value.items():
            if pid not in passage_length:
                passage_length[pid] = count
            else:
                passage_length[pid] += count

    return passage_length


def cal_tf_idf_passage(inverted_indices:Dict[str,Dict[str,int]], idf:Dict[str,float]) -> Dict[str,Dict[str,float]]:
    '''calculate the tf-idf for each term within the passage for each passage

    Input:
        inverted_indices: a dict whose keys are all terms in the vobavulary,\
            values are dicts with pids as keys and frequencies as values
        idf: a dict with terms as keys and idf as values

    Output:
        tf_idf: a dict whose keys are pids, values are dicts with all terms within the passage as keys\
            ,values are tf-idf value
    '''

    tf_idf = {}
    for term, fre4eachpas in inverted_indices.items():
        idf_term = idf[term]
        for pid,fre in fre4eachpas.items():
            if pid not in tf_idf.keys():
                tf_idf[pid] = {term: fre * idf_term}
            else:
                tf_idf[pid][term] = fre * idf_term

    print('finish : calculate tf-idf for passages')
    return tf_idf


def cal_tf_query(queries:pd.Series) -> List[Tuple[str,Dict[str,int]]]:
    '''calculate the frequency for each term within one quary
    
    Input:
        queries: (pid, list of tokens)

    Output:
        tf: a list of tuple (qid,value) where values are keys with all term within the query as keys and frequency as values
    '''

    tf = []
    for qid, tokens in queries.items():
        tf_dict = {}

        for token in tokens:
            if token not in tf_dict:
                tf_dict[token] = 1
            else:
                tf_dict[token] += 1

        tf.append((qid,tf_dict))

    print('finish : calculate tf for queries')

    return tf


def cal_tf_idf_query(tf:List[Tuple[str,Dict[str,int]]], idf:Dict[str,float]) -> List[Tuple[str,Dict[str,float]]]:
    '''calculate the tf-idf for each term within the query for each query

    Input:
        tf: a list of tuple (qid,value) where values are dicts with all term within the query as dicts and frequency as values
        idf: a dict with terms in vocabulary as keys and idf as values

    Output:
        tf_idf: a list of tuple (qid,value) where values are keys with all terms of the query as keys and tf-idf as values
    '''

    tf_idf = []

    for qid,tf_query in tf:
        tf_idf_dict = {}
        for term, tf_term in tf_query.items():
            if term not in idf:
                continue
            tf_idf_dict[term] = tf_term * idf[term]

        tf_idf.append((qid,tf_idf_dict))

    print('finish : calculate tf-idf for queries')

    return tf_idf


def cal_score_BM25(pid:str, inverted_indices:Dict[str,Dict[str,int]], df:dict, query_fre: dict, passage_length:int, ave_length: float, 
                   num_passages:int, k1:float, k2:float, b:float) -> float:
    '''calculate the score for BM25

    Input:
        pid: id of the passage
        inverted_indices: a dict representing inverted indicex. keys are all terms,\
            values are dicts with pid as keys and frequencies as values
        df: a dict with all terms in vocabulary as keys, the number of the passage containing the term as values
        query_fre: a dict with all terms in quary as indices, frequency within the query as values and qid as name
        passage_length: the length of the passage
        num_passage: number of the passages / size of the collection
        ave_length: average length of the passages
        k1, k2, b: hyparameters of the score function for BM25 

    Output:
        score: the BM25 socre of the pair (quary with qid and x with pid)
    '''

    K = k1 * ((1 - b) + b * passage_length / ave_length)

    score = 0
    for term, fre in query_fre.items():
        # skip terms of queires that are not in all passages
        if term not in inverted_indices:
            continue
        # skip terms of queires that are not in this passage
        if pid not in inverted_indices[term]:
            continue
        score += np.log(
            ((0 + 0.5) / (0 - 0 + 0.5)) / ((df[term] - 0 + 0.5 ) / (num_passages - df[term] - 0 + 0 + 0.5)) \
        * ((k1 + 1) * inverted_indices[term][pid]) / (K + inverted_indices[term][pid]) \
        * ((k2 + 1) * fre) / (k2 + fre)
        )

    return(score)


def retrieve_BM25(inverted_indices:dict, qid_pids:dict, tf_query:List[Tuple[str,Dict[str,int]]], 
                  filename:str,
                  k1:float, k2:float, b:float, print_detail = False):
    '''implement BM25

    Args:
        inverted_indices: a dict representing inverted indicex. keys are all terms in the vobavulary,\
            values are dicts with pid as keys and frequencies as values
        qid_pids: a dict with query as keys and the lists of the pids as values
        tf_query: a list of tuple (qid,value) where values are keys with all term within the query & vovabulary as keys and frequency as values
        filename: .csv file path to save the results 
        k1,k2,b: hyparameters of the score function for BM25 
        print_detail: True indicate printing if retrive is finish for each query

    Return:
        retrieve_all: retrieval results, (qid,pid,score)

    Output:
        a .csv file with (qid,pid,score) as rows, ordered by order of quary in tf_idf_quary, score DESC. For each qid it just keep top 100 pid
    '''
    print('start : BM25')
    df = {k:len(v) for k,v in inverted_indices.items()}
    passage_length = cal_length(inverted_indices)
    ave_length = np.average(list(passage_length.values()))
    num_passages = len(passage_length)

    retrieve_all = pd.DataFrame()
    # retrieve under each query
    for qid, query in tf_query:
        retrieve = {}
        for pid in qid_pids[qid]:
            # calculate score for each passage
            retrieve[pid] = cal_score_BM25(pid,inverted_indices,df,query, passage_length[pid],
                                           ave_length, num_passages,k1,k2,b)
        # concentate results of each query
        retrieve = sorted(retrieve.items(), key=lambda item: item[1], reverse=True)[0:100]
        retrieve = pd.DataFrame(retrieve)
        retrieve.insert(loc = 0, column= None, value = qid)
        retrieve_all = pd.concat([retrieve_all,retrieve], ignore_index=True)
        if print_detail:
            print('finish query {}'.format(qid))
    
    print('finfish : BM25')

    # save resutls
    retrieve_all.columns = ['qid','pid','score']
    retrieve_all.to_csv(filename,index=False,header=False)

    return retrieve_all


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


if __name__ == '__main__':

    df = pd.read_csv('validation_data.tsv',sep='\t',header=0)

    # process string to tokens
    pattern = re.compile(r'[^a-zA-Z\s]')
    nltk.download('stopwords')
    stopwords_eng = stopwords.words('english')
    funct_to_tokens = lambda line:to_tokens(line,pattern,stopwords_eng)

    df['queries'] = df['queries'].apply(funct_to_tokens)
    df['passage'] = df['passage'].apply(funct_to_tokens)

    # generate sub data
    queries = df[['qid','queries']]
    passages = df[['pid','passage']]
    qid_pid = df[['qid','pid']]
    qid_pid_relevance = df[['qid','pid','relevancy']]

    queries = queries.drop_duplicates('qid')
    queries = queries.set_index('qid')['queries']

    passages = passages.drop_duplicates('pid')
    passages = passages.set_index('pid')['passage']
    size = len(passages)

    qid_pid_group = qid_pid.groupby('qid')
    qid_pids_dict = {qid:pid['pid'].values for qid,pid in qid_pid_group}

    # calculate tf, idf, tf-idf for BM25
    inverted_indices = invert_index(passages)
    idf = cal_idf(inverted_indices=inverted_indices,num = size)
    tf_query = cal_tf_query(queries)
    tf_idf_query = cal_tf_idf_query(tf_query, idf)
    tf_idf_passage = cal_tf_idf_passage(inverted_indices, idf)

    results = retrieve_BM25(inverted_indices=inverted_indices,qid_pids=qid_pids_dict,filename='bm25.csv',
                  tf_query=tf_query,k1=1.2,k2=100,b = 0.75)
    
    # metrics
    map, ndcg = cal_metrics(results, qid_pid_relevance)

    print('the average precision for BM25 : {:.4f}, NDCG : {:.4f}'.format(map,ndcg))

    with open('metrics.txt','a',encoding='utf-8') as f:
        f.write(f'BM25 {map} {ndcg}\n')
