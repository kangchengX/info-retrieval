import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

import task1
import task2


def build_model(word_dim,num_class=1):
    '''build the model
    
    Args:
        word_dim: dimension of the word vector
        num_class: number of classes, 1 for (0.0,1,0) relevancy
    '''

    model = nn.Sequential(
        nn.Linear(word_dim, word_dim//2),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(word_dim//2, word_dim//4),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(word_dim//4,word_dim//8),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(word_dim//8,num_class),
        nn.Sigmoid()
    )

    return model


def train_model(data:pd.DataFrame,epoch:int,batch:int,lr:float):
    '''Args:
        data:(qid,pid,features,relevancy)
        epoch: number of epoches
        batch: batch size
        lr: learning rate
    '''

    word_dim = data['features'].iloc[0].shape[0]

    model = build_model(word_dim)

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
    results = task2.retrieve(data)

    return results


if __name__ == '__main__':
    
    df_train = pd.read_csv('data_train.csv')
    df_val = pd.read_csv('data_val.csv')
    df_train['features'] = np.load('data_train.npy',allow_pickle=True)
    df_val['features'] = np.load('data_val.npy',allow_pickle=True)

    model = train_model(df_train,10,1000,0.01)
    torch.save(model,'NN_model.pth')

    results = retrieve_nn(model,df_val)
    map,ndcg = task1.cal_metrics(results)

    print('the average precision for NN : {:.4f}, NDCG : {:.4f}'.format(map,ndcg))
    with open('metrics.txt','a',encoding='utf-8') as f:
        f.write(f'NN {map} {ndcg}\n')