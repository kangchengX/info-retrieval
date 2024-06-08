import numpy as np
import pandas as pd
import re
from tqdm.auto import tqdm
import gc
import utils
from functools import partial
from typing import Dict
import xgboost as xgb
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset

# track the  .apply on DataFrame
tqdm.pandas()

def to_vector(line:str, pattern:re.Pattern, word_embedding:Dict[str, np.ndarray]):
    '''tranform string to vector, by averaging the vectors for each token(term)
    
    Args:
        line: string to process
        pattern: pattern to get terms
        stop_words: stop words
        word_embedding: mapping from word to vector

    Returns:
        vector: mean word vector
    '''

    tokens = utils.generate_tokens(line=line, pattern=pattern)
    vectors = [word_embedding[word] for word in tokens if word in word_embedding]

    # generate zero vector if there is no embedding for any term of the line
    if len(vectors) == 0:
        return np.zeros(len(word_embedding['like']),dtype=word_embedding['like'].dtype)
    
    vector = np.mean(vectors,axis=0)

    return vector


def extract_features(df:pd.DataFrame,pattern:re.Pattern,word_embedding:Dict[str, np.ndarray]):
    '''extract features from the provided retrival results

    Args:
        df: (qid,pid,queries,passage,relevancy)
        pattern: pattern to get terms
        stop_words: stop words

    Return:
        df: (qid,pid,features,relevancy)
    '''

    to_vector_partial = partial(to_vector, 
                                pattern=pattern,
                                word_embedding=word_embedding)

    # generate vectors for queries and passage
    df['queries'] = df['queries'].progress_apply(to_vector_partial)
    gc.collect()
    df['passage'] = df['passage'].progress_apply(to_vector_partial)
    gc.collect()

    # contact the vectors for each (query, passage) pair
    queries_array = np.array(df['queries'].tolist())
    passage_array = np.array(df['passage'].tolist())

    df['features'] = np.concatenate((queries_array, passage_array),axis=1).tolist()

    df.drop(columns=['queries','passage'],inplace=True)

    gc.collect()

    return df


def _retrieve_learning_part(data:pd.DataFrame):
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


class LogisticRegression():
    '''The logistic regresson model
    '''

    def __init__(self,dim:int | None = 10) :
        '''initial the model
        
        Args: 
            dim: dimention of the feature
        '''

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


def train_lr(model:LogisticRegression,
            data:pd.DataFrame,
            epoch:int,
            batch_size:int,lr:float):
    '''train the logistic retrival model
    
    Args:
        model: the initialized model
        data: (qid,pid,'features',relevancy)
        epoch: number of epochs
        batch_size: size of batched

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

    results = _retrieve_learning_part(data)

    return results


def train_lm(data_train:pd.DataFrame,
             data_val:pd.DataFrame, 
             max_depth_range = range(5,8),
             n_estimators_range = [100,200,300]):
    '''train LambdaMART and tune the hyperperameters

    Args:
        data_train: to train the model on, (qid,pid,features,relevancy)
        data_val: to tune the hyperperameters on, (qid,pid,features,relevancy)
        max_depth_range: the range to choose max_depth from
        n_estimators_range: the range to choose n_estimators from
    '''

    # prepare inputs
    data_train = data_train.sort_values(by='qid')

    x_train = np.stack(data_train['features'].values)
    y_train= data_train['relevancy'].values

    group_counts = data_train.groupby('qid').size().values

    # metrics on the best model
    best_ndcg = 0.0
    best_map = 0.0

    # tune the hyperperparametes
    for max_depth in max_depth_range:
        for n_estimators in n_estimators_range:
            model = xgb.XGBRanker(
                    objective='rank:pairwise',
                    learning_rate=0.1,
                    max_depth=max_depth,
                    n_estimators=n_estimators
                )
            print('start : train the model')
            model.fit(x_train, y_train, group=group_counts)
            print('finish : train the model')

            # calculate metrics
            results = retrieve_lm(model,data_val)
            map,ndcg = utils.cal_metrics(results)

            if ndcg>best_ndcg and map>best_map:
                best_n_estimators = n_estimators
                best_max_depth = max_depth
                best_model = model

    return best_model, best_max_depth, best_n_estimators


def retrieve_lm(model:xgb.XGBRanker, data:pd.DataFrame):
    '''implement LambdaMART
    
    Args: 
        model: the trained LambdaMART Model
        data: DataFrame (qid,pid,features,relevancy)

    Returns:
        results: retrieval results, (qid,pid,score_lr,relevancy)
    '''
    
    x = np.stack(data['features'].values)
    data['score'] =  model.predict(x)
    data = data.drop(columns='features')

    results = _retrieve_learning_part(data)

    return results


class MLP(nn.Module):
    '''kkk'''
    def __init__(self, word_dim, num_class=1):
        '''Initialize the model
        
        Args:
            word_dim: dimension of the word vector
            num_class: number of classes, 1 for (0.0,1,0) relevancy
        '''
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(word_dim, word_dim//2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(word_dim//2, word_dim//4),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(word_dim//4, word_dim//8),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(word_dim//8, num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train_mlp(data:pd.DataFrame,epoch:int,batch:int,lr:float):
    '''Args:
        data:(qid,pid,features,relevancy)
        epoch: number of epoches
        batch: batch size
        lr: learning rate
    '''

    word_dim = data['features'].iloc[0].shape[0]

    model = MLP(word_dim)

    loss_object = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # perpare data
    x = np.stack(data['features'].values)
    y = data['relevancy'].values
    y = np.expand_dims(y,axis=-1)

    x = torch.tensor(x)
    y = torch.tensor(y,dtype=torch.float)

    dataset = TensorDataset(x,y)
    data_loader = DataLoader(dataset,batch_size=batch,shuffle=True)

    # train the model
    for i in range(epoch):
        for inputs, labels in data_loader:
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = loss_object(outputs, labels)  
            loss.backward()  
            optimizer.step() 

        print(f"Epoch {i+1}, Loss: {loss.item()}")

    return model


def retrieve_nn(model:nn.Module,data:pd.DataFrame):
    '''implement Neural Network Model
    
    Args: 
        model: the trained LambdaMART Model
        data: DataFrame (qid,pid,features,relevancy)

    Returns:
        results: retrieval results, (qid,pid,score_lr,relevancy)
    '''

    x = torch.tensor(np.stack(data['features'].values))

    # get the value
    model.eval()
    scores = (model.forward(x))
    data['score'] =  scores.detach().numpy()[:,0]

    data = data.drop(columns='features')
    results = _retrieve_learning_part(data)

    return results


# def to_files_feature(filename_train_open='train_data.tsv', filename_val_open='validation_data.tsv',
#                      filename_train_save='data_train',filename_val_save='data_val'):
#     '''extract features and save to .parquet files
    
#     Args:
#         filename_train_open: .tsv file containing training data, (qid,pid,queries,passage,relevancy)
#         filename_val_open: .tsv file containing validation data, (qid,pid,queries,passage,relevancy)
#         filename_train_save: .parquet file containing training features, (qid,pid,features,relevancy)
#         filename_train_save: .parquet file containing training features, (qid,pid,features,relevancy)
#     '''

#     print('start : load data')
#     df_train = pd.read_csv(filename_train_open,sep='\t',header=0, nrows=1500000)# use sub set because of the memory error
#     df_val = pd.read_csv(filename_val_open,sep='\t',header=0)
#     print('finish : load data')

#     pattern = re.compile(r'[^a-zA-Z\s]')
#     word2vec = downloader.load('word2vec-google-news-300')

#     nltk.download('stopwords')
#     stopwords_eng = stopwords.words('english')

#     print('start : extract_features')

#     print('training data : ')
#     df_train = extract_features(df_train,pattern,stopwords_eng,word2vec)
#     df_train[['pid','qid','relevancy']].to_csv(filename_train_save+'.csv', index=False)
#     np.save(filename_train_save+'.npy',df_train['features'].values)
#     del df_train

#     print('validation data : ')
#     df_val = extract_features(df_val,pattern,stopwords_eng,word2vec)
#     df_val[['pid','qid','relevancy']].to_csv(filename_val_save+'.csv', index=False)
#     np.save(filename_val_save+'.npy',df_val['features'].values)
#     del df_val

#     print('finish : extract_features')
