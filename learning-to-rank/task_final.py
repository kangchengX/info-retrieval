import task2,task3,task4
import pandas as pd
import xgboost as xgb
import torch
import re
from gensim import downloader
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords


def results_save(df:pd.DataFrame,method:str):
    '''save the retrived results
    
    Args:
        df: ('qid','pid','score')
        filename: file to save the results

    Output:
        files contaning the retrived results
    '''

    df['assignment'] = 'A2'
    df['method'] = method

    df['rank'] = df.groupby('qid').cumcount() + 1

    df = df[['qid','assignment','pid','rank','score','method']]

    df.to_csv(method+'.txt',sep=' ',header=None,index=None)


if __name__ == '__main__':

    df = pd.read_csv('candidate_passages_top1000.tsv',sep='\t',header=None)
    df.columns = ['qid','pid','queries','passage']

    pattern = re.compile(r'[^a-zA-Z\s]')
    word2vec = downloader.load('word2vec-google-news-300')

    nltk.download('stopwords')
    stopwords_eng = stopwords.words('english')

    df = task2.extract_features(df,pattern,stopwords_eng,word2vec)

    ## LR
    model = task2.LogisticRegression()
    model.load('LR_model.npy')
    results = task2.retrieve_lr(model,df)
    results_save(results,'LR')

    ## LM
    model = xgb.XGBRanker()
    model.load_model('LM_model.json')
    results = task3.retrieve_lm(model,df)
    results_save(results,'LM')

    # NN
    model = torch.load('NN_model.pth')
    results = task4.retrieve_nn(model,df)
    results_save(results,'NN')
