import re
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time

def tokens_process(line:str, pattern:re.Pattern, lemmatizer, stopwords = [], vocabulary = None):
    '''process the line into list of lemmatized tokens

    Input: 
        line: the string to process
        pattern: the Pattern in re
        lemmatizer: the lemmatizer in nltk.stem to lemmatize the tokens
        stopwords: list of the words to remove
        vocabulary: None or set

    Output:
        tokens: a list of the processed tokens
    '''

    tokens = pattern.sub(' ',line).lower().split()
    tokens = [token for token in tokens if token not in stopwords]
    tokens_tags = nltk.pos_tag(tokens)
    proccessed_tokens = []
    for token, tag in tokens_tags:
        if tag.startswith('J'):
            pos = 'a'
        elif tag.startswith('V'):
            pos = 'v'
        elif tag.startswith('R'):
            pos = 'r'
        else:
            pos = 'n'
        token = lemmatizer.lemmatize(token,pos)
        proccessed_tokens.append(token)

    if vocabulary is not None:
        proccessed_tokens = [token for token in proccessed_tokens if token in vocabulary]

    return proccessed_tokens


def cal_tf_passage(filename:str, stopwords: list, lemmatizer) -> Tuple[Dict[str,Dict[str,int]],dict,set]:
    '''calculate the frequency for each term within the passage & not stop words for each passage

    Input: 
        filename: path of the query-passage file
        lemmatizer: the lemmatizer in nltk.stem to lemmatize the tokens
        stopwords: list of the words to remove
        
    Output:
        tf_passage: a dict whose keys are pids, values are dicts with terms as keys and frequencies as values
        query_candidate: a dict with pids as keys and lists of candicate passages as values
        vocabulary: a set of the lemmatized terms 
    '''

    print("start : calculate terms' frequences for each passage from " + filename)

    pattern = re.compile(r'[^a-zA-Z\s]')
    tf_passage = {}
    query_candidate = {}
    vocabulary = set()

    with open(filename,'r',encoding='utf-8') as f:
        for item in f:
            details = item.split('\t')
            qid = details[0]
            pid = details[1]
            passage = details[3]

            # get the pids for each qid
            if qid not in query_candidate.keys():
                query_candidate[qid] = [pid]
            else:
                query_candidate[qid].append(pid)

            # count each term's frequency within one passage
            if pid in tf_passage.keys():
                continue
            tokens = tokens_process(passage,pattern,lemmatizer,stopwords)
            tokens_count = {}
            for token in tokens:
                vocabulary.add(token)
                if token not in tokens_count:
                    tokens_count[token] = 1
                else:
                    tokens_count[token] += 1
            tf_passage[pid] = tokens_count

    print("finish : calculate terms' frequences for each passage from " + filename)

    return tf_passage, query_candidate, vocabulary

def cal_score_discounting(query:list, passage:dict, v:int, eps) -> float:
    '''calculate the log score under discounting smoothing

    Input:
        query: a list of terms within the query & vocabulary (not unique)
        passage: a dict with terms within the passage & vocabulary as keys and counts as values
        v: number of unique words in the entire collection
        eps: parammter for the smoothing method.

    Output:
        score: the log score
    '''

    score = 0
    d = sum(passage.values())

    for term in query:
        if term not in passage:
            score += np.log(eps /(eps * v+d))
        else:
            score += np.log((eps + passage[term]) / (eps * v+d))

    return score

def cal_score_dirichlet(query:list, passage:dict, tf_collection:dict, mu, length_collection) -> float:
    '''calculate the log score under dirichlet smoothing

    Input:
        query: a list of terms within the query & vocabulary (not unique)
        passage: a dict with terms within the passage & vocabulary as keys and counts as values
        tf_collection: a dict with terms in vocabulary as keys and counts as values
        mu: parammter for the smoothing method.

    Output:
        score: the log score
    '''

    score = 0
    d = sum(passage.values())
    lam = d / (d + mu)

    for term in query:
        if term not in passage:
            score += np.log((1-lam) * tf_collection[term] / length_collection)
        else:
            score += np.log(lam * passage[term] / d + (1-lam) * tf_collection[term] / length_collection)

    return score

def get_queries(filename:str , stopwords: list, lemmatizer, vocabulary:set):
    '''get the queries from the test queries file
    
    Input:
        filename: path for the test queries file
        stopwords: the words to remove
        lemmatizer: to lemmatize the tokens
        vocabulary: a set containing all lemmatized terms in vocabulary
    
    Output:
        queries: a list of tuple(qid,tokens) where tokens are listed of the terms (not unique) within the query & not the stop words
    '''

    print('start : get queries from ' + filename)

    queries = []
    patten = re.compile(r'[^a-zA-Z\s]')
    with open(filename,'r',encoding='utf-8') as f:
        content = f.readlines()

    for item in content:
        item = item.split('\t')
        qid = item[0]
        query = item[1]
        tokens = tokens_process(query, patten, lemmatizer, stopwords, vocabulary)
        queries.append((qid,tokens))

    print('finish : get queries from ' + filename)

    return queries

def retrieve_query_liklihood(queries:list, tf_passage:dict, query_candidate:dict, 
                             smooth_type:str, pram,filename_out:str, print_details = False):
    '''implement the query_liklihood retreival model
    
    Input:
        queries: a list of tuple(qid,tokens) where tokens are listed of the terms (not unique) within the query
        tf_passage: a dict whose keys are pids values are dicts with terms as keys and frequencies as values
        query_candidate: a dict with pids as keys and lists of cadicate passages as values
        smooth_type: type of the smoothing method
        pram: prameter for the smoothing fuction
        filename_out: the path of the .csv where the results are stored

    Output:
        (not return): a .csv file with qid,pid,score as rows
    '''
    print('start : query liklihood retrieve using ' + smooth_type)

    retrieve_all = pd.DataFrame()

    # calculate the frequencies of terms in the entire collection
    tf_collection = {}
    for terms in tf_passage.values():
        for term, count in terms.items():
            if term not in tf_collection.keys():
                tf_collection[term] = count
            else:
                tf_collection[term] += count
    
    v = len(tf_collection)

    if smooth_type == 'discount':
        funct = lambda x: cal_score_discounting(x[0],x[1],v,pram)
    elif smooth_type == 'interp':
        funct = lambda x: cal_score_dirichlet(x[0],x[1],tf_collection,pram,sum(tf_collection.values()))
    else:
        print('unknown smothing method')
        return
    
    # sort the resultes for each qid and get the top100 passages
    for qid, query_terms in queries:
        retrieve = {}
        for pid in query_candidate[qid]:
            retrieve[pid] = funct([query_terms,tf_passage[pid]])
        retrieve = sorted(retrieve.items(), key=lambda item: item[1],reverse=True)[0:100]
        retrieve = pd.DataFrame(retrieve)
        retrieve.insert(loc = 0, column= None, value = qid)
        retrieve_all = retrieve_all.append(retrieve, ignore_index=True)
        if print_details:
            print('finish query {}'.format(qid))
    
    print('finish : query liklihood retrieve using ' + smooth_type)
    
    retrieve_all.to_csv(filename_out,index=False,header=False)

def cal_diff_passage(filename1:str, filenmam2:str):
    '''calculate the difference between the results of two retrieval methods by comparing the retrieved passages
    
    Input: 
        filename1, filenmae2: the files to compare

    output
        score: the similarity score
    '''

    df1 = pd.read_csv(filename1,encoding='utf-8',header=None,names=['qid','pid','score'])
    df2 = pd.read_csv(filenmam2,encoding='utf-8',header=None,names=['qid','pid','score'])

    qids = set(df1['qid'].unique())
    scores = []
    # compare pids under each qid
    for qid in qids:
        df1_pid = df1[df1['qid'] == qid]['pid']
        df2_pid = df2[df2['qid'] == qid]['pid']
        score = len(set(df1_pid).intersection(set(df2_pid))) / (len(df1_pid))
        scores.append(score)

    score = np.average(scores)

    return score


if __name__ == '__main__':
    
    t1 = time.time()

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    stopwords_eng = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # reterive using three smoothing methods
    smoothing_list = [('discount','laplace.csv',1), 
                ('discount','lidstone.csv',0.1),
                ('interp','dirichlet.csv',50)]

    tf_passage, query_candidate, vocabulary = cal_tf_passage('candidate-passages-top1000.tsv',stopwords=stopwords_eng, lemmatizer=lemmatizer)
    queries = get_queries('test-queries.tsv',stopwords_eng,lemmatizer,vocabulary)

    for smooth_type, filename, pram in smoothing_list:
        retrieve_query_liklihood(queries=queries,tf_passage=tf_passage,query_candidate=query_candidate,smooth_type=smooth_type,
                                pram=pram,filename_out=filename)
        
    for filename in ['laplace.csv','lidstone.csv', 'dirichlet.csv']:
        df = pd.read_csv(filename,encoding='utf-8',header=None,names=['qid','pid','score'])
        print(filename + ' ave score = {}'.format(np.average(df['score'])))

    passage_length = [sum(v.values()) for v in tf_passage.values()]
    print('average_passage_length is {}'.format(np.average(passage_length)))

    t2 = time.time()
    print('the total time : {}'.format(t2-t1))
