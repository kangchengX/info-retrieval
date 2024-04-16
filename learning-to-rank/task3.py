import xgboost as xgb
import pandas as pd
import numpy as np
import task1, task2
import os

# requiremnet for xgb.XGBRanker
try:
    import sklearn
except ImportError:
    print('sklearn is not installed.')
    os.system('pip install scikit-learn')


def train_lm(data_train:pd.DataFrame,data_val:pd.DataFrame, 
             max_depth_range = range(5,8),n_estimators_range = [100,200,300]):
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
            map,ndcg = task1.cal_metrics(results)

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

    results = task2.retrieve(data)

    return results


if __name__ == '__main__':
    
    print('start : load data')
    df_train = pd.read_csv('data_train.csv')
    df_val = pd.read_csv('data_val.csv')
    df_train['features'] = np.load('data_train.npy',allow_pickle=True)
    df_val['features'] = np.load('data_val.npy',allow_pickle=True)
    print('finish : load data')

    model,max_depth,n_estimators = train_lm(df_train,df_val,range(5,8),[100,200,300])
    model.save_model('LM_model.json')

    results = retrieve_lm(model,df_val)
    map,ndcg = task1.cal_metrics(results)
    print('the average precision for LambdaMART : {:.4f}, average NDCG : {:.4f}'.format(map,ndcg))
    print('best max_depth : {}, best n_estimators : {}'.format(max_depth,n_estimators))
    with open('metrics.txt','a',encoding='utf-8') as f:
        f.write(f'LM {map} {ndcg} {max_depth} {n_estimators} \n')
