import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple
import xgboost as xgb
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from warnings import warn


class RetrieveBaseModel:
    '''The base model to train. Polymorphism for the predict method.'''
    def predict(self):
        '''Predict.'''
        raise NotImplementedError("Subclass must implement abstract method")


class LogisticRegression(RetrieveBaseModel):
    '''The logistic regression model

    Attributes:
        w : the weights
    '''

    def __init__(self, dim: int | None = 10) :
        '''initial the model
        
        Args: 
            dim: dimention of the feature
        '''

        self.w = np.zeros(dim)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        '''Forward process of the model.

        Args:
            x: inputs
        
        Returns:
            y: the forward outputs
        '''
        assert inputs.shape[1] == self.w.shape[0]
        y = np.dot(inputs, self.w)
        y = 1 / (1 + np.exp(-y))

        return y
    
    def backward(self, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, lr: float):
        '''Backword process. Calculate the gradient and update weights
        
        Args:
            x: the inputs
            y_true: the true labels (or targets)
            y_pred: the model outputs
            lr: the learning rate    
        '''
        assert x.shape[0] == y_true.shape[0] == y_pred.shape[0]
        dw = np.dot(x.T, (y_pred-y_true)) / x.shape[0]
        self.w -= lr * dw

    def cal_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        '''Calculate loss according to the outputs and labels
        
        Args:
            y_true: the true labels (or targets)
            y_pred: the model outputs

        Returns:
            loss: the loss value
        '''
        offset = 1e-5
        loss = -np.mean(y_true * np.log(y_pred+offset) + (1 - y_true) * np.log(1 - y_pred+offset))
        return loss
    
    def save(self, filename: str):
        '''Save the Logistic Model
        
        Args:
            filename: the filename to save the model
        '''
        np.save(filename, self.w)

    
    def load(self, filename: str):
        '''Load the model
        
        Args:
            filename: the filename to load the model
        '''
        self.w = np.load(filename)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        '''Get the prediction of the model
        
        Args:
            inputs: inputs of the model
        '''
        return self.forward(inputs)
    

class LambdaMART(xgb.XGBRanker, RetrieveBaseModel):
    '''The LambdaMART Retrieval Model'''
    def __init__(self, max_depth: int, n_estimators: int, **kwargs):
        '''Initialize the model
        
        Args:
            learning_rate, max_depth, n_estimators
        '''
        super().__init__(
            objective='rank:pairwise', 
            max_depth=max_depth,
            n_estimators=n_estimators,
            **kwargs
        )
    

class MLP(nn.Module, RetrieveBaseModel):
    '''The simple mlp model'''
    def __init__(self, word_dim: int, num_class: int | None = 1):
        '''Initialize the model
        
        Args:
            word_dim: dimension of the word vector
            num_class: number of classes, Default is 1 for (0.0,1,0) relevancy
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        '''Forward process of the model.

        Args:
            x: inputs
        
        Returns:
            the forward outputs
        '''
        return self.model(inputs)    
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        '''Predict
        
        Args:
            inputs: inputs of the model
        '''
        self.to('cpu')
        tensor = torch.tensor(inputs, dtype=next(self.parameters()).dtype)
        self.eval()
        
        with torch.no_grad():  # No gradient calculation for inference
            outputs = self.forward(tensor)
        
        return outputs.numpy()[:,0]


class Trainer:
    '''The class to train the model
    
    Attributes:
        model: the model to train
        data: the data to train on
        train_lag: indicates if the model has been trained
    '''
    def __init__(
        self,
        model: Union[LogisticRegression, MLP, LambdaMART],
        data: pd.DataFrame
    ):
        '''Initialize the model
        
        Args:
            model: the model to train
            data: the data to train on
        '''
        self.model = model
        self.data = data
        self.train_lag = False

    def get_model(self) -> Union[LogisticRegression, MLP, LambdaMART]:
        '''Get the 'trained' model'''
        if not self.train_lag:
            warn('the model has not been trained yet')
        return self.model
    
    def reset_model(self, model: Union[LogisticRegression, MLP, LambdaMART]):
        '''Reset the model to train'''
        self.model = model
        self.train_lag = False

    def train(self, **kwargs):
        '''Train the model'''
        self.train_lag = True
        if isinstance(self.model, LogisticRegression):
            self.train_lr(**kwargs)
        elif isinstance(self.model, MLP):
            self.train_mlp(**kwargs)
        elif isinstance(self.model, LambdaMART):
            self.train_lm(**kwargs)
        else:
            raise AttributeError('Unsupported model class')

    def _prepare_inputs_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Prepare inputs and labels
        
        Returns: (inputs, labels)
            inputs: inputs of the model
            labels: labels of the model
        '''
        inputs = np.stack(self.data['features'].values)
        labels = self.data['relevancy'].values

        return inputs, labels


    def train_lr(self, batch_size: int, epoch: int, learning_rate: float):
        '''Train the logistic regression model
        
        Args: 
            batch_size: the batch size
            epoch: number of epochs
            learning_rate: learning rate 

        Returns:
            losses_epoch: list of losses for all epochs
        '''

        print('start: train the Logistic Model')
        inputs, labels = self._prepare_inputs_labels()

        size = len(labels)
        batch_num = size // batch_size
        batch_last = size % batch_size

        losses_epoch = []

        # start training
        for i in range(epoch):
            indices = np.random.choice(size,size,replace=False) # shuffle the data
            losses_batch = []
            # train for each batch
            for j in range(batch_num):
                indices_batch = indices[j*batch_size:(j+1)*batch_size]
                outputs = self.model.forward(inputs[indices_batch])
                losses_batch.append(self.model.cal_loss(labels[indices_batch],outputs))
                self.model.backward(inputs[indices_batch],labels[indices_batch],outputs,learning_rate)
            # deal with the remaining data
            if batch_last != 0:
                indices_batch = indices[-batch_last:]
                outputs = self.model.forward(inputs[indices_batch])
                losses_batch.append(self.model.cal_loss(labels[indices_batch],outputs[indices_batch]))
                self.model.backward(inputs[indices_batch],labels[indices_batch],outputs,learning_rate)

            print('Epoch : {}, Loss : {:.4f}'.format(i+1,np.mean(losses_batch)))
            losses_epoch.append(np.mean(losses_batch))

        print('finish: train the Logistic Model')

        return losses_epoch

    def train_lm(self, learning_rate: float | None = None):
        '''Train the LambdaMART Model
        
        Args:
            learning_rate: learning rate of the training. If None, use the default learning rate
        '''

        print('start: train the LambdaMART model')
        if learning_rate is not None:
            self.model.set_params(learning_rate=learning_rate)
        # prepare data
        data_train = self.data.sort_values(by='qid')
        x_train = np.stack(data_train['features'].values)
        y_train= data_train['relevancy'].values
        group_counts = data_train.groupby('qid').size().values

        self.model.fit(x_train, y_train, group=group_counts)

        print('finish: train the LambdaMART model')


    def train_mlp(self, batch_size, epoch, learning_rate):
        '''Train the mlp model 
        
        Args:
            batch_size: the batch size
            epoch: number of epochs
            learning_rate: learning rate 

        Returns:
            losses_epoch: list of losses for all epochs
        '''

        print('start: train the mlp model')
        loss_object = nn.BCELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # perpare data
        x = np.stack(self.data['features'].values)
        y = self.data['relevancy'].values

        y = np.expand_dims(y,axis=-1)
        x = torch.tensor(x)
        y = torch.tensor(y,dtype=torch.float)

        dataset = TensorDataset(x,y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses_epoch = []

        # train the model
        for i in range(epoch):
            losses_batch = []
            for inputs, labels in data_loader:
                optimizer.zero_grad()  
                outputs = self.model(inputs)  
                loss = loss_object(outputs, labels)  
                loss.backward()  
                optimizer.step() 
                losses_batch.append(loss.item())

            print(f"Epoch {i+1}, Loss: {np.mean(losses_batch)}")
            losses_epoch.append(losses_batch)

        print('finish: train the mlp model')
        return losses_epoch


class LearningRetriever:
    def __init__(
        self, 
        model: RetrieveBaseModel,
        data: pd.DataFrame
    ):
        """Initialize the model
        
        Args:
            model: the retrieval model
            data: the data to retrieve on
        """
        
        self.model = model
        self.data = data

    def retrieve(self, num_top_results: int | None = None):
        '''Args: '''
        print('start: retrieval model')
        score_df = self._calculate_score()
        results = self._retrieve_from_score(score_df, num_top_results=num_top_results)
        print('finish: retrieval model')

        return results

    def reset_model(self, model: RetrieveBaseModel):
        '''reset the model
        
        Args: 
            model: the model
        '''

        self.model = model

    def _calculate_score(self):
        x = np.stack(self.data['features'].values)
        score_df = self.data[['pid','qid','relevancy']]
        score_df['score'] = self.model.predict(x)

        return score_df
    
    def _retrieve_from_score(self, score_df: pd.DataFrame, num_top_results: int | None = None):
        '''Generate retrieval results. Get the num_top_results for each query if num_top_results is not None.
        
        Args:
            score_df: (qid,pid,score,relevancy)
            num_top_results: If not None, get the num_top_results for each query

        return: 
            results: retrieval results, (qid,pid,score,relevancy)
        '''

        data_group = score_df.groupby('qid',sort=False)

        results = pd.DataFrame()
        for _, passage in data_group:
            scores = passage.sort_values(by='score',ascending=False)[0:num_top_results]
            results = pd.concat([results, scores], ignore_index=True)

        return results
    