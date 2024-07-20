import numpy as np
import pandas as pd
import re
from functools import partial
from utils import generate_tokens, Lemmatizer
from typing import Iterable, Literal, Dict, Callable, Union
from tqdm.auto import tqdm
from warnings import warn
from gensim.models import KeyedVectors

# track the  .apply of tokenizer on DataFrame
tqdm.pandas()

class DocPidDictView:
    '''The class to create a dict view of the original dict at a doc (or passage) level given the pid. 
    The dict view can be seen as a dict of a doc (or passage) with structure {term: value}, where term is the term of this 
    doc (or passage), and the value is the doc-level (or passage-level) feature for the term. 
    Specifically, term is the key of the original dict whose ['pids_dict'] has the assigned 'pid', and value is original[term]['pids_dict'][pid][key]
        
    Atrributes:
        original: the original dict
        key: the key of the deepest dictionary
        pid: the passage id      
    '''
    def __init__(
            self, 
            feature_name: Literal['tf', 'tf-unnorm', 'tf-idf'],
            original: Dict[str, Dict[str, Dict[str | int, Dict[str, int | float]]]] | None = {},
            pid: str | int | None = None,
            ignore_pid_init_error: bool | None = False
    ):
        """Initilaize the class
        
        Args:
            feature_name: the passage-level feature name for the term
            original: the original dict with the structure:
                Dict[str, Dict[str, Dict[str | int, Dict[str, int | float]]]], i.e.,
                {
                    tokens, {
                                pids_dict : {
                                                pids, {
                                                          tf : float,
                                                          tf-unnorm : int,
                                                          tf-idf : float
                                                      }
                                            },
                            }
                }
                , default is {}
            pid: the pid of the dict. Default is None
            ignore_pid_init_error: If False, an error will be raised if pid is not assigned a value in initialization
        """
        self.original = original
        self.feature_name = feature_name
        if not ignore_pid_init_error and pid is None:
            raise ValueError(f'pid is not assigned a value')
        self.pid = pid

    def update_pid(self, pid: str | int):
        '''Updata the pid. After updating, the view can be seen as a new dict of the passage with a simple structure {term: value}'''
        self.pid = pid

    def __getitem__(self, term: str):
        '''Get the passage-level feature value of the term
        
        Returns: the passage-level feature value
        '''
        return self.original[term]['pids_dict'][self.pid][self.feature_name]

    def get(self, term: str, default: int | float | None = None):
        '''Get the passage-level feature value of the term. If the original dict doesn't have the term, or
        self.original[term]['pids_dict'] doesn't have the given pid, return default
        
        Returns: the passage-level feature value or the default
        '''
        return self.original.get(term, {}).get('pids_dict', {}).get(self.pid, {}).get(self.feature_name, default)
        
    def __iter__(self):
        '''Iter the dict view keys. This is depreciated, since the structure of the original dict, i.e., 'inverted indices' is
        designed for getting the passage-level feature value from terms at a collection-level efficiently. And the set of keys
        to access are determined from another variable (like a query) for a high efficiency.

        Returns: set of terms whose 'pids_dict' has the given pid
        '''
        
        warn('This is depreciated. Please use a sperate set from this dict to determine the keys to access')
        return (term for term in self.original if self.pid in self.original[term]['pids_dict'])

    def keys(self):
        '''Get the dict view keys (i.e. term of the passage with the assigned pid). This is depreciated, 
        since the structure of the original dict i.e. 'inverted indices' is
        designed for getting the passage-level feature value from terms at a collection-level efficiently. And the set of keys
        to access are determined from another variable (like a query) for a high efficiency.

        Returns: set of term of the passage with the assigned pid
        '''

        warn('This is depreciated. Please use a sperate set from this dict to determine the keys to access')
        return (term for term in self.original if self.pid in self.original[term]['pids_dict'])

    def items(self):
        '''Get the dict view keys and values. This is depreciated, since the structure of the original dict i.e. 'inverted indices' is
        designed for getting the passage-level feature value from terms at a collection-level efficiently. And the set of keys
        to access are determined from another variable (like a query) for a high efficiency.

        Returns: iter of (term, value) where term is the term of the passage with the assigned pid, value is the corresponding passage-level feature value
        '''

        warn('This is depreciated. Please use a sperate set from this dict to determine the keys to access')
        for term, feature_collection_dict in self.original.items():
            pids_dict = feature_collection_dict['pids_dict']
            if self.pid in pids_dict:
                yield term, pids_dict[self.pid][self.feature_name]

    def values(self):
        '''Get the dict view keys and values. This is depreciated, since the structure of the original dict i.e. 'inverted indices' is
        aimed for getting the passage-level feature value from the terms in a collection-level efficiently. And the set of keys
        to assess are determined from another variable (like a query)

        Returns: iter passage-level feature value of the given pid
        '''

        warn('This is depreciated. Please use a sperate set from this dict to determine the terms to use')
        for feature_collection_dict in self.original.values():
            pids_dict = feature_collection_dict['pids_dict']
            if self.pid in pids_dict:
                yield pids_dict[self.pid][self.feature_name]

    def __contains__(self, term: str):
        '''Allows for 'in' operator. Check if the passage with the assigned pid have the given term'''

        return term in self.original and self.pid in self.original[term]['pids_dict']


class OneNestedDictView:
    '''The class to create a dict view of a one-nested dict. The dict view can be seen as {term, value},
    where term is the key of the original dict, an the value is original[term][inner_key]
    
    Attributes:
        original: the original key with shape:
            Dict[str, Dict[str, int | float]], i.e.
            {
                tokens, {   
                            inner keys, int | float
                        }
            }
        key: the feature_name
    '''
    def __init__(
            self, 
            original: Dict[str, Dict[str, int | float]],
            inner_key: int
    ):
        '''Initialize the class
        
        Args:
            original: the original key with shape:
                Dict[str, Dict[str: int | float]], i.e.
                {
                    keys, {   
                               inner keys, int | float
                          }
                }
                
            inner_key: the key for the inner (deepest) dict
        '''
        self.original = original
        self.inner_key = inner_key

    def updata_original(self, original: Dict[str, Dict[str, int | float]]):
        '''Update the original dict
        
        Args:
            original: the original key with shape:
                Dict[str, Dict[str: int | float]], i.e.
                {
                    keys, {   
                               inner keys, int | float
                          }
                }
        '''
        self.original = original

    def __getitem__(self, key: str):
        '''Get the inner value
        
        Returns: the inner value
        '''
        return self.original[key][self.inner_key]
    
    def __iter__(self):
        '''Iter the keys of the original, i.e., the dict view's keys
        
        Returns: iter of the keys
        '''
        return iter(self.original)

    def get(self, key: str, default: int | float | None = None):
        '''Get the inner value. Return default if the orginal dict doesn't have the key, i.e.,
        the dict view doesn't have the key
        
        Returns: the inner value or default
        '''
        
        return self.original.get(key, {}).get(self.inner_key, default)

    def keys(self):
        '''Get keys of the original, i.e. the dict view's keys
        
        Return: keys of the original dict
        '''
        return self.original.keys()

    def items(self):
        '''Iter the keys of the original dict and inner values
        
        Returns: Iter of tuples (key, inner value)
        '''
        for key, inner_dict in self.original.items():
            yield key, inner_dict[self.inner_key]

    def values(self):
        '''Iter the inner values
        
        Returns: Iter of inner values
        '''
        for inner_dict in self.original.values():
            yield inner_dict[self.inner_key]

    def __contains__(self, key: str):
        '''Allows for 'in' operator'''

        return key in self.original


class DocCollectionDictView(OneNestedDictView):
    '''A dict view at the collection-level of the inverted indices
    
    Attributes:
        original: the docs dict with shape:
            Dict[str, Dict[str, int | float]], i.e.
            {
                tokens, {   
                            {
                                'idf': float,
                                'tf-collection': int
                            }
                        }
            }
        feature_name: the feature name
    '''
    def __init__(
            self, 
            feature_name: Literal['idf', 'tf-collection'],
            original: Dict[str, Dict[str, int | float]] | None = {}
        ):
        '''Initialize the view class
        
        Args:
            feature_name: the feature name
            original: the docs dict with shape:
                Dict[str, Dict[str: int | float]], i.e.
                {
                    tokens, {   
                                {
                                    'idf': float,
                                    'tf-collection': int
                                }
                            }
                }
        '''
        super().__init__(original=original, inner_key=feature_name)

class QueryDictView(OneNestedDictView):
    '''A dict view of the query for the assigned feature
    
    Attributes:
        original: the query dict with shape:
            Dict[str, Dict[str: int | float]], i.e.
            {
                tokens, {   
                            {
                                'tf': float,
                                'tf-unnorm': int,
                                'tf-idf': float
                            }
                        }
            }
        feature_name: the feature name
    '''

    def __init__(
            self,
            feature_name: Literal['tf', 'tf-unnorm', 'tf-idf'],
            original: Dict[str, Dict[str, int | float]] | None = {}
    ):
        '''Initialize the view class
        
        Args:
            feature_name: the feature name
            original: the docs dict with shape:
                Dict[str, Dict[str: int | float]], i.e.
                {
                    tokens, {   
                                {
                                    'tf': float,
                                    'tf-unnorm': int,
                                    'tf-idf': float
                                }
                            }
                }
        '''
        super().__init__(original=original, inner_key= feature_name)


class DocLoader():
    '''Class to load docs (or passages)

    Attributes:
        to_tokens: fuction to tokenize the string
        num_docs: number of documents (or passage)
        inverted_indices: A complex dict with the structure:
            Dict[str, Dict[str, Union[Dict[Union[str,int], Dict[str, Union[int,float]]], float]]], i.e.,
            {
                tokens, {
                            pids_dict : {
                                            pids, {
                                                     'tf': float,
                                                     'tf-unnorm': int,
                                                     'tf-idf': float
                                                  }
                                        },
                            idf : float,
                            tf-collection : int
                        }
            }
        passages_length: dict with pids as keys and length as values
        average_length: average length of the passages (docs)

    Methods:
        load: load the documents (or passages)
    '''

    to_tokens: Callable
    num_docs: int | None
    inverted_indices: Dict[str, Dict[str, Union[Dict[Union[str,int], Dict[str, Union[int,float]]], float]]]
    passages_length: dict | None
    average_length: int | None

    def __init__(
            self,
            pattern : re.Pattern | None = re.compile(r'[^a-zA-Z\s]'), 
            lemmatizer: Lemmatizer | None = None, 
            stopwords : Iterable | None = None, 
            vocabulary : Iterable | None = None
    ):
        """Initialize the class
        
        Args:
            pattern: the Pattern in re. Default is re.compile(r'[^a-zA-Z\\s]').
            lemmatizer: the lemmatizer to lemmatize the tokens. If None, not lemmatize.
                Default is None
            stopwords: words to remove. If None, keep all tokens.
                Default is None
            vocabulary: words to keep. If none, keep all tokens.
                Default is None
        """

        self.to_tokens = partial(
            generate_tokens, 
            pattern=pattern, 
            lemmatizer=lemmatizer, 
            stopwords=stopwords, 
            vocabulary=vocabulary
        )
        
        self.num_docs = None
        self.inverted_indices = {}
        self.passages_length = None
        self.average_length = None

    def load(self, raw_data: pd.DataFrame):
        '''Load the docs (or passages) into inverted indices
        
        Args:
            raw_data: queries and passages (or documents), with row (pid,qid,query,passage)
        '''
        # generate Series with pid as index and passage as values
        raw_data_series = raw_data.drop_duplicates('pid').set_index('pid')['passage']
        # convert passage string to list of tokens for each passage
        pids_tokens = raw_data_series.progress_apply(self.to_tokens)

        # calculate statistics for the collection
        self.num_docs = len(pids_tokens)
        self.passages_length = pids_tokens.apply(len).to_dict()
        self.average_length = np.mean(list(self.passages_length.values()))

        # generate the inverted indices
        print('start : calculate inverted indices')
        self._calculate_tf(pids_tokens)
        self._calculate_tf_idf()
        print('finish : calculate inverted indices')
        
    def _calculate_tf(self, pid_tokens: pd.Series):
        '''For each term, calculate tf, unnormalized tf for the passage and the unnormalized frequency for the entire collection
        
        Args:
            pid_tokens: pd.Series with pids as indices and lists of tokens as values
        '''

        for pid, tokens in pid_tokens.items():
            tokens_unique, counts = np.unique(tokens, return_counts=True)
            for token, count in zip(tokens_unique, counts):
                feature_collection_dict = self.inverted_indices.setdefault(token, {'pids_dict': {}, 'tf-collection': 0})
                feature_collection_dict['tf-collection'] += count

                feature_collection_dict['pids_dict'][pid] = {
                    'tf' : count / self.passages_length[pid],
                    'tf-unnorm' : count
                }

    def _calculate_tf_idf(self):
        '''Calculate the idf for each term for the whole collection
        as well as tf-idf for each term for the passage.
        '''

        for feature_collection_dict in self.inverted_indices.values():
            pids_dict = feature_collection_dict['pids_dict']
            # calculate inverted_indices[token]['idf']
            feature_collection_dict['idf'] = np.log(self.num_docs / len(pids_dict))
            # calculate inverted_indices[token]['pids_dict']['pid']['tf-idf']
            for feature_passage_dict in pids_dict.values():
                feature_passage_dict['tf-idf'] = feature_passage_dict['tf'] * feature_collection_dict['idf']


class QueryLoader():
    """The query loader
    
    Attributes:
        self.to_tokens: the tokenizer
        self.queries: A complex dict, with the following structure:
            Dict[Union[str, int], Dict[str, Dict[str, Union[int, float]]]], i.e.,
            {
                qids, {
                          tokens, {
                                     'tf': float,
                                     'tf-unnorm': int,
                                     'tf-idf': float
                                  }
                      }
            }

    Methods:
        load: load the queries
    """

    to_tokens: Callable
    queries: Dict[Union[str, int], Dict[str, Dict[str, Union[int, float]]]]

    def __init__(
            self, 
            pattern: re.Pattern | None = re.compile(r'[^a-zA-Z\s]'), 
            lemmatizer: Lemmatizer | None = None, 
            stopwords: Iterable | None = None, 
            vocabulary: Iterable | None = None
    ):
        """Initialize the class
        
        Args:
            pattern: the Pattern in re. Default is re.compile(r'[^a-zA-Z\\s]').
            lemmatizer: the lemmatizer to lemmatize the tokens. If None, not lemmatize.
                Default is None
            stopwords: words to remove. If None, keep all tokens.
                Default is None
            vocabulary: words to keep. If none, keep all tokens.
                Default is None
        """
        self.to_tokens = partial(
            generate_tokens, 
            pattern=pattern, 
            lemmatizer=lemmatizer, 
            stopwords=stopwords, 
            vocabulary=vocabulary
        )
        self.queries = None
        self._calculate_tf_idf_query_partial = None

    def load(self, doc_loader: DocLoader, raw_data: pd.DataFrame):
        '''Load the queries and the corresponding statistics
        
        Args:
            doc_loader: the doc_loader containing the doc (or passages)
            raw_data: raw_data containing the columns ('qid','pid','query','passage')
        '''
        print('start : calculate queries statistics')
        self._calculate_tf_idf_query_partial = partial(
            self._calculate_tf_idf_query_part,
            doc_loader = doc_loader)
        queries_series = raw_data.drop_duplicates(subset='qid').set_index('qid')['query']
        self._calculate_tf_idf(queries_series, doc_loader)
        print('finish : calculate queries statistics')
    
    def _calculate_tf_idf(self, queries_series: pd.Series, doc_loader: DocLoader):
        '''Calcualte tf, unnormalized tf, tf-idf of each term under each query'''
        # calculate queries['qid'][token]['tf']
        self.queries = queries_series.progress_apply(self.to_tokens).apply(self._calculate_tf_idf_query).to_dict()
    
    def _calculate_tf_idf_query_part(self, query_tokens: list, doc_loader: DocLoader):
        '''Calcualte tf, unnormalized tf, tf-idf of each term for this query'''
        tokens_unique, counts = np.unique(query_tokens, return_counts=True)
        return {
            token : {
                'tf' : count / len(query_tokens),
                'tf-unnorm': count,
                'tf-idf': doc_loader.inverted_indices.get(token,{}).get('idf', 0) 
            }
            for token, count in zip(tokens_unique, counts)
        }
    
    def _calculate_tf_idf_query(self, query_tokens: list):
        return self._calculate_tf_idf_query_partial(query_tokens)


class DataLoader():
    """The loader to load the queries and documents (or passages)
    
    Attribute:
        doc_loader (DocLoader): loader of the documents (or passages)
        query_loader (QueryLoader): loader of the queris

    Method:
        load: load the data
    """
    def __init__(self, **kwargs):
        """Initialize the class
        
        Args:
            pattern: the Pattern in re. Default is re.compile(r'[^a-zA-Z\\s]').
            lemmatizer: the lemmatizer in nltk.stem to lemmatize the tokens. If None, not lemmatize.
                Default is None
            stopwords: words to remove. If None, keep all tokens.
                Default is None
            vocabulary: words to keep. If none, keep all tokens.
                Default is None
        """
        self.doc_loader = DocLoader(**kwargs)
        self.query_loader = QueryLoader(**kwargs)
        self.query_candidate = None
        self.lemmatizer = kwargs.get('lemmatizer', None)

    def load(self, raw_data: pd.DataFrame):
        '''Load the data set as well as statistics'''
        self.query_candidate = raw_data[['qid', 'pid']]
        self.doc_loader.load(raw_data)
        self.query_loader.load(doc_loader=self.doc_loader, raw_data=raw_data)
        if self.lemmatizer is not None:
            self.lemmatizer.clear_cache()


def extract_passage_embedding(doc_loader: DocLoader, word_embedding: KeyedVectors):
    '''Extract passage vector embedding for each passage
    
    Args:
        doc_loader: the loader containing the passages
        word_embedding: a object mapping the word to the vector

    Returns:
        passage_embedding: DataFrame with columns ('pid', 'vector')
    '''
    inverted_indices = doc_loader.inverted_indices
    passages_length = doc_loader.passages_length
    zero_vector = np.zeros(len(word_embedding['like']), dtype=word_embedding['like'].dtype)
    passage_embedding = {}

    # iter each term
    for term, feature_collection_dict in inverted_indices.items():
        word_vector = word_embedding[term] if term in word_embedding else zero_vector
        for pid, feature_passage_dict in feature_collection_dict['pids_dict'].items():
            passage_embedding[pid] = passage_embedding.get(pid, zero_vector) + feature_passage_dict['tf-unnorm'] * word_vector
    
    # average the vectors
    passage_embedding = {pid: sum_vector / passages_length[pid] 
                         for pid, sum_vector in passage_embedding.items()}
    # convert dict to DataFrame
    passage_embedding = pd.DataFrame(list(passage_embedding.items()), columns=['pid', 'passage_vector'])
    
    return passage_embedding


def extract_query_embedding(query_loader: QueryLoader, word_embedding: KeyedVectors):
    '''Extract query vector embedding for each query
    
    Args:
        query_loader: the loader containing the queries
        word_embedding: an object mapping the word to the vector

    Returns:
        query_embedding: DataFrame with columns ('qid', 'vector')
    '''
    zero_vector = np.zeros(len(word_embedding['like']), dtype=word_embedding['like'].dtype)
    queries = query_loader.queries
    query_embedding =  {
        qid: np.mean(
            [
                word_embedding[term] * feature_dict['tf-unnorm']
                if term in word_embedding else zero_vector
                for term, feature_dict in term_dict.items()
            ],
            axis=0
        )
        for qid, term_dict in queries.items()
    }
    
    query_embedding = pd.DataFrame(list(query_embedding.items()), columns=['qid','query_vector'])

    return query_embedding


def extract_features(
        data_loader: DataLoader,
        word_embedding: KeyedVectors,
        query_candidate: pd.DataFrame,
        filename: str | None = None,
        concatenate: bool | None = True
):
    """Extract feature representation for the query-passage pair. Use inverted indices in data_loader
    to improve efficiency. 
    
    Args:
        data_loader: the data loader containing the passages and queris
        word_embedding: an object mapping the word to the vector
        query_candidate: DataFrame which mush include columns (qid, pid)
        filename: filename to save the features. If None, not save the features
        concatenate: If true, concatenate the query vector and the passage_vector

    Returns:
        features: dataframe with columns ('qid','pid','features') if concatenate,
            with columns ('qid', 'pid', 'query_vector', 'passage_vector' if not concatenate),
            where each row of 'features', 'query_vector', or 'passage_vector' is a 1-d array
    """
    query_candidate = query_candidate[['qid','pid']]
    passage_embedding = extract_passage_embedding(data_loader.doc_loader, word_embedding)
    query_embedding = extract_query_embedding(data_loader.query_loader, word_embedding)

    features = query_candidate.merge(passage_embedding, on='pid').merge(query_embedding, on='qid')
    
    if concatenate:
        vectors = np.concatenate(
            (np.stack(features['query_vector'].values, axis=0), 
             np.stack(features['passage_vector'].values, axis=0)), 
            axis = 1
        )
        features['features'] = list(vectors)
        features = features[['qid','pid','features']]

    if filename is not None:
        features.to_csv(filename)

    return features