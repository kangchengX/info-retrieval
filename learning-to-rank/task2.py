import numpy as np
import pandas as pd
import re
import os
from gensim import downloader
from typing import Iterable
import matplotlib.pyplot as plt
import nltk
import warnings
from nltk.corpus import stopwords
from tqdm.auto import tqdm
import task1
import gc

# track the  .apply on DataFrame
tqdm.pandas()


class LogisticRegression():
    def __init__(self,dim:int | None = 10) :
        self.w = np.zeros(dim)
    
    def forward(self,x:np.ndarray):
        assert x.shape[1] == self.w.shape[0]
        y = np.dot(x, self.w)
        y = 1 / (1 + np.exp(-y))

        return y
    
    def backward(self,x:np.ndarray,y_true:np.ndarray,y_pred:np.ndarray,lr:float):
        assert x.shape[0] == y_true.shape[0] == y_pred.shape[0]
        dw = np.dot(x.T, (y_pred-y_true)) / x.shape[0]
        self.w -= lr * dw

    def cal_loss(self, y_true,y_pred):
        offset = 1e-5
        loss = -np.mean(y_true * np.log(y_pred+offset) + (1 - y_true) * np.log(1 - y_pred+offset))
        return loss
    
    def save(self,filename):
        np.save(filename,self.w)

    
    def load(self,filename):
        self.w = np.load(filename)



def model_train(model:LogisticRegression,data:pd.DataFrame,
                epoch:int,batch_size:int,lr:float):
    '''train the model
    
    Args:
        model: the initialized model
        data: (qid,pid,'features',relevancy)

    Returns:
        loss_epoch: loss for each epoch
    '''

    inputs = np.stack(data['features'].values)
    labels = data['relevancy'].values

    size = len(labels)

    batch_num = size // batch_size
    batch_last = size % batch_size

    loss_epoch = []

    print('start : train model')
    for i in range(epoch):
        indices = np.random.choice(size,size,replace=False)
        loss = []
        for j in range(batch_num):
            indices_batch = indices[j*batch_size:(j+1)*batch_size]
            outputs = model.forward(inputs[indices_batch])
            loss.append(model.cal_loss(labels[indices_batch],outputs))
            model.backward(inputs[indices_batch],labels[indices_batch],outputs,lr)

        if batch_last != 0:
            indices_batch = indices[-batch_last:]
            outputs=model.forward(inputs[indices_batch])
            loss.append(model.cal_loss(labels[indices_batch],outputs[indices_batch]))
            model.backward(inputs[indices_batch],labels[indices_batch],outputs,lr)

        print('Epoch : {}, Loss : {:.4f}'.format(i+1,np.mean(loss)))
        loss_epoch.append(np.mean(loss))

    print('finish : train model')

    return loss_epoch

def retrieve(data:pd.DataFrame):
    '''get the top 100 results for each query
    
    Args:
        data: (qid,pid,score,relevancy)

    return: 
        results: retrieval results, (qid,pid,score,relevancy)
    '''

    data_group = data.groupby('qid',sort=False)
    results = pd.DataFrame()
    for _,passage in data_group:
        scores = passage.sort_values(by='score',ascending=False)
        scores = scores[0:100]
        results = pd.concat([results, scores], ignore_index=True)

    return results


def retrieve_lr(model:LogisticRegression, data:pd.DataFrame):
    ''' Logistic Regression retrieval model
    
    Args: 
        model: the trained regression model
        data: DataFrame (qid,pid,features,relevancy)

    Returns:
        results: retrieval results, (qid,pid,score_lr,relevancy)
    '''

    data['score'] =  model.forward(np.stack(data['features'].values))
    data = data.drop(columns='features')

    results = retrieve(data)

    return results


def tokens_to_vector(tokens:list,word2vec):
    '''transform tokens to vectors

    Args: 
        tokens: list of string
        word2vect: mapping from word to vector

    Returns:
        vector: the tokens' vector
    '''

    vectors = [word2vec[word] for word in tokens if word in word2vec]
    if len(vectors) == 0:
        return np.zeros(len(word2vec['like']),dtype=word2vec['like'].dtype)
    vector = np.mean(vectors,axis=0)

    return vector


def to_vector(line:str,pattern:re.Pattern,stop_words:Iterable,word2vec):
    '''tranform string to vector
    
    Args:
        line: string to process
        pattern: pattern to get terms
        stop_words: stop words
        word2vect: mapping from word to vector

    Returns:
        vector: mean word vector
    '''

    tokens = task1.to_tokens(line,pattern,stop_words)
    vector = tokens_to_vector(tokens,word2vec)

    return vector


def extract_features(df:pd.DataFrame,pattern:re.Pattern,stop_words:Iterable,word2vec):
    '''
    Args:
        df: (qid,pid,queries,passage,relevancy)

    Return:
        df: (qid,pid,features,relevancy)
    '''

    funct_to_vector = lambda tokens:to_vector(
        tokens,pattern=pattern,stop_words=stop_words, word2vec=word2vec)

    df['queries'] = df['queries'].progress_apply(funct_to_vector)
    gc.collect()
    df['passage'] = df['passage'].progress_apply(funct_to_vector)
    gc.collect()

    funct_to_features = lambda row:np.concatenate([row['queries'],row['passage']])

    df['features'] = df.progress_apply(funct_to_features,axis=1)

    df.drop(columns=['queries','passage'],inplace=True)

    gc.collect()

    return df


def loss_plot(losses:list,styles:list,filename:str):
    '''plot loss across differenct learning rates
    
    Args:
        losses: list of tuple(learning rate, losses for each epoch)
        styles: list of plotting styles
        filename: filename to save the plot

    Output:
        an image of loss plotting
    '''

    assert len(losses) == len(styles)

    fig, ax = plt.subplots(figsize = (9, 6))

    for loss, style in zip(losses,styles):
        lr,loss_values = loss
        ax.plot(range(1,1+len(loss_values)),loss_values,
                style,markersize=5,markerfacecolor='none',
                label = str(lr))

    ax.set_title('changes of loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    plt.xticks(range(1,1+len(loss_values)))

    plt.savefig(filename)
    plt.show()


def to_files_feature(filename_train_open='train_data.tsv', filename_val_open='validation_data.tsv',
                     filename_train_save='data_train',filename_val_save='data_val'):
    '''extract features and save to .parquet files
    
    Args:
        filename_train_open: .tsv file containing training data, (qid,pid,queries,passage,relevancy)
        filename_val_open: .tsv file containing validation data, (qid,pid,queries,passage,relevancy)
        filename_train_save: .parquet file containing training features, (qid,pid,features,relevancy)
        filename_train_save: .parquet file containing training features, (qid,pid,features,relevancy)
    '''

    print('start : load data')
    df_train = pd.read_csv(filename_train_open,sep='\t',header=0, nrows=1500000)# use sub set because of the memory error
    df_val = pd.read_csv(filename_val_open,sep='\t',header=0)
    print('finish : load data')

    pattern = re.compile(r'[^a-zA-Z\s]')
    word2vec = downloader.load('word2vec-google-news-300')

    nltk.download('stopwords')
    stopwords_eng = stopwords.words('english')

    print('start : extract_features')

    print('training data : ')
    df_train = extract_features(df_train,pattern,stopwords_eng,word2vec)
    df_train[['pid','qid','relevancy']].to_csv(filename_train_save+'.csv', index=False)
    np.save(filename_train_save+'.npy',df_train['features'].values)
    del df_train

    print('validation data : ')
    df_val = extract_features(df_val,pattern,stopwords_eng,word2vec)
    df_val[['pid','qid','relevancy']].to_csv(filename_val_save+'.csv', index=False)
    np.save(filename_val_save+'.npy',df_val['features'].values)
    del df_val

    print('finish : extract_features')


if __name__ == '__main__':

    # to_files_feature()

    df_train = pd.read_csv('data_train.csv')
    df_val = pd.read_csv('data_val.csv')
    df_train['features'] = np.load('data_train.npy',allow_pickle=True)
    df_val['features'] = np.load('data_val.npy',allow_pickle=True)

    print(df_train.head())
    print(df_val.head())

    dim = df_train['features'].iloc[0].shape[0]
    model = LogisticRegression(dim)
    
    loss = model_train(model,df_train,20,1000,0.01)

    model.save('LR_model.npy')
    losses = [(0.01,loss)]

    results = retrieve_lr(model,df_val)
    map, ndcg = task1.cal_metrics(results)

    # compare the difference of learning rates
    for lr in [0.001,0.0001]:
        model = LogisticRegression(dim)
        loss = model_train(model,df_train,20,1000,lr)
        losses.append((lr,loss))

    styles = ['r-','b-','g-']

    print('the average precision for Logistic Regression (lr = 0.01) : {:.4f}, NDCG : {:.4f}'.format(map,ndcg))
    with open('metrics.txt','a',encoding='utf-8') as f:
        f.write(f'LR {map} {ndcg}\n')

    # show loss
    loss_plot(losses,styles,'loss-lr.png')
