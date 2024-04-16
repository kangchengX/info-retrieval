import numpy as np
import pandas as pd
import re
from typing import Tuple,Dict,List
import task1
import task2
import time

def cal_idf(inverted_indices:Dict[str,Dict[str,int]], num:int) -> Dict[str,int]:
    ''' calculate the idf value of the terms in vocabulary

    Input:
        inverted_indices: a dict whose keys are all terms in the vobavulary,\
            values are dicts with pids as keys and frequencies as values
        num: number of the passages (size the collection)

    Output:
        idf : a dict with terms as keys and idf as values
    '''

    idf = {k:np.log10(num/len(v)) for k,v in inverted_indices.items()}
    print('finish : calculate idf')

    return idf


def cal_tf_idf_passage(inverted_indices:Dict[str,Dict[str,int]], idf:Dict[str,float]) -> Dict[str,Dict[str,float]]:
    '''calculate the tf-idf for each term within the passage for each passage

    Input:
        inverted_indices: a dict whose keys are all terms in the vobavulary,\
            values are dicts with pids as keys and frequencies as values
        idf: a dict with terms as keys and idf as values

    Output:
        tf_idf: a dict whose keys are pids, values are dicts with all terms within the passage&vocabulary as keys\
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


def cal_tf_query(filename:str,vocabulary:set) -> List[Tuple[str,Dict[str,int]]]:
    '''calculate the frequency for each term within one quary
    
    Input:
        filename: path of the query file
        vocabulary: a set containing all terms in the vocabulary

    Output:
        tf: a list of tuple (qid,value) where values are keys with all term within the query & vovabulary as keys and frequency as values
    '''
 
    tf = [] # use list instead of dict to better preserve the order of quary
    pattern = re.compile(r'[^a-zA-Z\s]')
    with open(filename,'r',encoding='utf-8') as f:
        content = f.readlines()

    for item in content:
        item = item.split('\t')
        qid = item[0]
        quary = item[1]

        tokens = pattern.sub(' ',quary).lower().split()
        tf_dict = {}

        for token in tokens:
            if token not in vocabulary:
                continue
            if token not in tf_dict.keys():
                tf_dict[token] = 1
            else:
                tf_dict[token] += 1

        tf.append((qid,tf_dict))

    print('finish : calculate tf for queries')

    return tf

def cal_tf_idf_query(tf:List[Tuple[str,Dict[str,int]]], idf:Dict[str,float]) -> List[Tuple[str,Dict[str,float]]]:
    '''calculate the tf-idf for each term within the query for each query

    Input:
        tf: a list of tuple (qid,value) where values are dicts with all term within the query & vocabulary as dicts and frequency as values
        idf: a dict with terms in vocabulary as keys and idf as values

    Output:
        tf_idf: a list of tuple (qid,value) where values are keys with all terms of the query & vocabulary as keys and tf-idf as values
    '''

    tf_idf = []

    for qid,tf_query in tf:
        tf_idf_dict = {}
        for term, tf_term in tf_query.items():
            tf_idf_dict[term] = tf_term * idf[term]

        tf_idf.append((qid,tf_idf_dict))

    print('finish : calculate tf-idf for queries')

    return tf_idf

def cal_score_tfidf(tfidf_query:dict, tfidf_passage:dict) -> np.float64:
    '''calulate the score for TF-IDF vector space based retrieval model

    Input:
        tfidf_query: a dict with terms within the query & vocabulary as keys and tf-idf as values
        tfidf_passage: a dict with terms within the passage & vocabulary as keys and tf-idf as values
    
    Output:
        score: the cosine of the two vectors
    '''
    product = sum(tfidf_query[key] * tfidf_passage[key] for key in tfidf_query.keys() if key in tfidf_passage.keys())
    score = product / np.linalg.norm(list(tfidf_query.values())) / np.linalg.norm(list(tfidf_passage.values()))
    return score

def retrieve_vector_space(tf_idf_passage:Dict[str,Dict[str,float]], tf_idf_quary:List[Tuple[str,Dict[str,float]]],
                          query_candidate:dict, filename:str, print_details = False) -> None:
    '''implement the TF-IDF vector space based retrieval model

        Input: 
            tf_idf_passage: a dict whose keys are pids,\
                values are dicts with all terms within the passage & vocabulary as keys,\
                values are tf-idf value
            tf_idf_quary: a list of tuple (qid,value),\
                where values are keys with all terms of the quary & vocabulary as keys\
                and tf-idf as values
            query_candidate: a dict with query as keys and the lists of the pids as values
            filename: the path of the .csv file storing the results of the model
            print_details: True indicate printing if retrive is finish for each query

        Output:
            (not return): a .csv file with (qid,pid,score) as rows, ordered by order of quary in tf_idf_quary, score DESC. For each qid it just keep top 100 pid
    '''
    print('start : TF-IDF vector space based retrieval model')
    retrieve_all = pd.DataFrame()
    for qid, row_query in tf_idf_quary:
        retrieve = {}
        for pid in query_candidate[qid]:
            row_passage = tf_idf_passage[pid]
            retrieve[pid] = cal_score_tfidf(row_query, row_passage)

        retrieve = sorted(retrieve.items(), key=lambda item: item[1], reverse=True)[0:100]
        retrieve = pd.DataFrame(retrieve)
        retrieve.insert(loc = 0, column= None, value = qid)
        retrieve_all = retrieve_all.append(retrieve, ignore_index=True)
        if print_details:
            print('finish query {}'.format(qid))
    
    retrieve_all.to_csv(filename,index=False,header=False)
    print('finish : TF-IDF vector space based retrieval model. Length is {}'.format(len(retrieve_all.index)))

def cal_length(inverted_indices:Dict[str,Dict[str,int]]) -> dict:
    '''calculate the length of each passage
    
    Input:
        inverted_indices: a dict representing inverted indices. keys are all terms in the vobavulary,\
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

def cal_score_BM25(pid:str, inverted_indices:Dict[str,Dict[str,int]], df:dict, query_fre: dict, passage_length:int, ave_length: float, 
                   num_passages:int, k1:float, k2:float, b:float) -> float:
    '''calculate the score for BM25

    Input:
        pid: id of the passage
        inverted_indices: a dict representing inverted indicex. keys are all terms in the vobavulary,\
            values are dicts with pid as keys and frequencies as values
        df: a dict with all terms in vocabulary as keys, the number of the passage containing the term as values
        query_fre: a dict with all terms in quary & vocabulary as indices, frequency within the quary as values and qid as name
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
        if pid not in inverted_indices[term]:
            continue
        score += np.log(
            ((0 + 0.5) / (0 - 0 + 0.5)) / ((df[term] - 0 + 0.5 ) / (num_passages - df[term] - 0 + 0 + 0.5)) \
        * ((k1 + 1) * inverted_indices[term][pid]) / (K + inverted_indices[term][pid]) \
        * ((k2 + 1) * fre) / (k2 + fre)
        )

    return(score)


def retrieve_BM25(inverted_indices:dict, query_candidate:dict, filename:str, 
                  tf_query:List[Tuple[str,Dict[str,int]]], k1:float, k2:float, b:float, print_detail = False):
    '''implement BM25

    Input:
        inverted_indices: a dict representing inverted indicex. keys are all terms in the vobavulary,\
            values are dicts with pid as keys and frequencies as values
        query_candidate: a dict with query as keys and the lists of the pids as values
        filename: .csv file path to save the results 
        tf_query: a list of tuple (qid,value) where values are keys with all term within the query & vovabulary as keys and frequency as values
        k1,k2,b: hyparameters of the score function for BM25 
        print_detail: True indicate printing if retrive is finish for each query

    Output:
        (not return) a .csv file with (qid,pid,score) as rows, ordered by order of quary in tf_idf_quary, score DESC. For each qid it just keep top 100 pid
    '''
    print('start : BM25')
    df = {k:len(v) for k,v in inverted_indices.items()}
    passage_length = cal_length(inverted_indices)
    ave_length = np.average(list(passage_length.values()))
    num_passages = len(passage_length)

    retrieve_all = pd.DataFrame()
    for qid, query in tf_query:
        retrieve = {}
        for pid in query_candidate[qid]:
            retrieve[pid] = cal_score_BM25(pid,inverted_indices,df,query, passage_length[pid],
                                           ave_length, num_passages,k1,k2,b)
        retrieve = sorted(retrieve.items(), key=lambda item: item[1], reverse=True)[0:100]
        retrieve = pd.DataFrame(retrieve)
        retrieve.insert(loc = 0, column= None, value = qid)
        retrieve_all = pd.concat([retrieve_all,retrieve], ignore_index=True)
        if print_detail:
            print('finish query {}'.format(qid))
    
    print('finfish : BM25')
    
    retrieve_all.to_csv(filename,index=False,header=False)

if __name__ == '__main__':
    # tf-idf retrieve
    t1 = time.time()

    _,fre = task1.extract_frequencies('passage-collection.txt',True)
    inverted_indices, size, query_candidate = task2.invert_index('candidate-passages-top1000.tsv', set(fre.keys()))
    idf = cal_idf(inverted_indices=inverted_indices,num = size)
    tf_query = cal_tf_query('test-queries.tsv',set(fre.keys()))
    tf_idf_query = cal_tf_idf_query(tf_query, idf)
    tf_idf_passage = cal_tf_idf_passage(inverted_indices, idf)

    retrieve_vector_space(tf_idf_passage,tf_idf_query,query_candidate,'tfidf.csv')

    t2 = time.time()
    print('time for tf-idf retrieve is {}'.format(t2-t1))

    # BM25 retreive
    retrieve_BM25(inverted_indices=inverted_indices,query_candidate=query_candidate,filename='bm25.csv',
                  tf_query=tf_query,k1=1.2,k2=100,b = 0.75)
    
    t3 = time.time()
    print('time for BM25 retrieve is {}'.format(t3-t2))