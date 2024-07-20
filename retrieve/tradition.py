from typing import Literal
from warnings import warn
from functools import partial
import numpy as np
import pandas as pd
from data import DataLoader, QueryDictView, DocPidDictView, DocCollectionDictView


class Scorer:
    '''The class to calculate the retrieval score for the traditional methods
    
    Attributes:
        data_loader: the data loader containing the passages and queries
        kwargs: the dict with hyperperameters of the score function as keys
        score_type: type of the score function to use

        tf_query_view: a dict veiw mapping the query term to the tf value
        tf_passage_view: a dict view mapping the passage term to the tf value
        tf_idf_query_view: a dict view mapping the query term to the tf-idf value
        tf_idf_passage_view: a dict view mapping the passage term to the tf-idf value
        tf_collection_view: a dict view mapping the terms to the tf of the entire collection

        _calculate_score_tfidf_partial, if score_type == 'tf-idf',
        _calculate_score_BM25_partial, if score_type == 'bm25',
        _calculate_score_discounting_partial, if score_type == 'laplace' or 'lidstone',
        _calculate_score_dirichlet_partial, if score_type == 'dirichlet',
        
        score_funct: the score function
        
    Methods:
        calcualte_score: calculate the score given pid and qid
        reset_score_type: reset the method to calcualte the score
    '''

    def __init__(
            self,
            data_loader: DataLoader,
            score_type: Literal['tf-idf', 'bm25', 'laplace', 'lidstone', 'dirichlet'],
            **kwargs
    ):
        """Initialize the model.
        
        Args:
            data_loader: the data loader containing the queries and docs (or passages)
            score_type: the method to calculate the score function.Can be:
                'tf-idf': use doc product of tf-idf vectors
                'bm25': score of the BM25
                'laplace': log likelihood with Laplace smoothing
                'lidstone': log likelihood with Laplace smoothing and Lidstone correction
                'dirichlet': log likelihood with Dirichlet smoothing
            kwargs: the hyperamters for the score function, can be:
                'k1' (float), 'k2' (float), 'b' (float), for 'bm25', Default is 1.2, 100, 0.75 respectively
                'eps' (float), for 'lidstone', Default is 1 for 'laplace'
                'mu' (float), for 'dirichlet', Default is 50
        """
        self.data_loader = data_loader
        self.score_type = score_type
        self.kwargs = kwargs

        self._add_dict_views()
        self._add_score_function_partial()
        self._add_score_function()
        
    def __call__(self, qid: str, pid: str):
        '''call the calcualte_score'''
        return self.calcualte_score(qid=qid, pid=pid)
    
    def calcualte_score(self, qid: str, pid: str):
        '''Calculate the socre
        
        Args:
            qid: qid of the query
            pid: pid of the passage

        Returns: 
            score
        '''
        return self.score_funct(qid=qid, pid=pid)
    
    def reset_score_type(
            self,
            score_type: Literal['tf-idf', 'bm25', 'laplace', 'lidstone', 'dirichlet'],
            **kwargs
    ):
        '''Reset the score type
        
        Args:
            data_loader: the data loader containing the queries and docs (or passages)
            score_type: the method to calculate the score function.Can be:
                'tf-idf': use doc product of tf-idf vectors
                'bm25': score of the BM25
                'laplace': log likelihood with Laplace smoothing
                'lidstone': log likelihood with Laplace smoothing and Lidstone correction
                'dirichlet': log likelihood with Dirichlet smoothing
            kwargs: the hyperamters for the score function, can be:
                'k1' (float), 'k2' (float), 'b' (float), for 'bm25', Default is 1.2, 100, 0.75 respectively
                'eps' (float), for 'lidstone', Default is 1 for 'laplace'
                'mu' (float), for 'dirichlet', Default is 50
        '''
        self.score_type = score_type
        self.kwargs = kwargs
        # # set the 'mu' as 1 by default
        # # by doing this, the 'mu' can be fixed when partial calculate_score_discounting
        # self.kwargs['mu'] = self.kwargs.get('mu', 1)

        if score_type == 'lidstone' and np.isclose(self.kwargs.get('mu', 50), 1):
            warn(f'mu is close to 1')

        self._add_score_function_partial()
        self._add_score_function()

    def _add_dict_views(self):
        '''Add dict views'''
        self.tf_query_dict_view = QueryDictView(feature_name='tf')
        self.tf_passage_dict_view = DocPidDictView(
            feature_name='tf',
            original=self.data_loader.doc_loader.inverted_indices,
            ignore_pid_init_error=True
        )
        self.tf_idf_query_dict_view = QueryDictView(feature_name='tf-idf')
        self.tf_idf_passage_dict_view = DocPidDictView(
            feature_name='tf-idf',
            original=self.data_loader.doc_loader.inverted_indices,
            ignore_pid_init_error=True
        )
        
        self.tf_collection_dict_view = DocCollectionDictView(
            feature_name='tf-collection',
            original=self.data_loader.doc_loader.inverted_indices
        )

    def _add_score_function_partial(self):
        '''Add the corresponding partial score function according to the score type
        to fix the hyperperameters of the score functions as well as collection-level features.'''
        if self.score_type == 'bm25':
            self._calculate_score_BM25_partial = partial(
                calculate_score_BM25, 
                tf_collection=self.tf_collection_dict_view,
                ave_length=self.data_loader.doc_loader.average_length,
                num_passages=self.data_loader.doc_loader.num_docs,
                **self.kwargs
            )

        elif self.score_type == 'laplace' or self.score_type == 'lidstone':
            self._calculate_score_discounting_partial = partial(
                calculate_score_discounting,
                num_unique_words=len(self.data_loader.doc_loader.inverted_indices),
                **self.kwargs
            )
        
        elif self.score_type == 'dirichlet':
            self._calculate_score_dirichlet_partial = partial(
                calculate_score_dirichlet,
                tf_collection=self.tf_collection_dict_view,
                length_collection = self.data_loader.doc_loader.num_docs,
                **self.kwargs
            )

        elif self.score_type != 'tf-idf':
            raise ValueError(f'Unknown score_type {self.score_type}')
            
    def _add_score_function(self):
        '''Add the corresponding score function according to the score type'''
        if self.score_type == 'tf-idf':
            self.score_funct = self._calculate_score_tfidf
        elif self.score_type == 'bm25':
            self.score_funct = self._calculate_score_BM25
        elif self.score_type == 'laplace' or self.score_type == 'lidstone':
            self.score_funct = self._calculate_score_discounting
        elif self.score_type == 'dirichlet':
            self.score_funct = self._calculate_score_dirichlet


    def _calculate_score_tfidf(self, qid, pid):
        '''Calculate score for tf idf vector space '''
        self.tf_idf_query_dict_view.updata_original(self.data_loader.query_loader.queries[qid])
        self.tf_idf_passage_dict_view.update_pid(pid)

        return calculate_score_tfidf(
            tf_idf_query=self.tf_idf_query_dict_view,
            tf_idf_passage=self.tf_idf_passage_dict_view
        )
    
    def _calculate_score_BM25(self, qid, pid):
        '''Calculate score for BM25'''
        self.tf_query_dict_view.updata_original(self.data_loader.query_loader.queries[qid])
        self.tf_passage_dict_view.update_pid(pid)
        
        return self._calculate_score_BM25_partial(
            tf_query=self.tf_query_dict_view,
            tf_passage=self.tf_passage_dict_view,
            passages_length=self.data_loader.doc_loader.passages_length[pid]
        )
    
    def _calculate_score_discounting(self, qid, pid):
        '''Calculate score for likelihood model with Laplace smoothing with or without Lidstone correction'''
        self.tf_query_dict_view.updata_original(self.data_loader.query_loader.queries[qid])
        self.tf_passage_dict_view.update_pid(pid)
        
        return self._calculate_score_discounting_partial(
            tf_query=self.tf_query_dict_view,
            tf_passage=self.tf_passage_dict_view,
            passages_length=self.data_loader.doc_loader.passages_length[pid]
        )
    
    def _calculate_score_dirichlet(self, qid, pid):
        '''Calculate score for likelihood model with dirichlet smoothing'''
        self.tf_query_dict_view.updata_original(self.data_loader.query_loader.queries[qid])
        self.tf_passage_dict_view.update_pid(pid)

        return self._calculate_score_dirichlet_partial(
            tf_query=self.tf_query_dict_view,
            tf_passage=self.tf_passage_dict_view,
            passages_length=self.data_loader.doc_loader.passages_length[pid]
        )
    

class TraditionRetriever():
    def __init__(
            self,
            data_loader: DataLoader,
            retrieve_type: Literal['tf-idf', 'bm25', 'likelihood'],
            smooth_type: Literal['laplace', 'lidstone', 'dirichlet'] | None = None,
            **kwargs
    ):
        """Initialize the class
        
        Args:
            data_loader: the data loader containing the queries and docs (or passages)
            retrieve_type: the retrieval type. Can be 'tf-idf', 'bm25' or 'likelihood'
            smooth_type: the smoothing method to use when retrieve_type is 'likelihood'
                Can be 'laplace', 'lidstone', 'dirichlet'
            kwargs: the hyperamters for the score function, can be:
                'k1' (float), 'k2' (float), 'b' (float), for 'bm25', Default is 1.2, 100, 0.75 respectively
                'eps' (float), for 'discounting', Default is 1
                'mu' (float), for 'dirichlet', Default is 50
        """
        self.retrieve_type = retrieve_type
        self.kwargs = kwargs
        self.smooth_type = smooth_type
        score_type = self._check_determine_score_type()
        self._add_message()
        self.scorer = Scorer(data_loader=data_loader,
                             score_type=score_type,
                             **self.kwargs)
        

    def _add_message(self):
        '''Add the message to print to the Attribute message'''
        if self.retrieve_type == 'tf-idf':
            self.message = 'TF-IDF vector space based retrieval model'
        elif self.retrieve_type == 'bm25':
            self.message = 'BM25 retrieval model'
        elif self.retrieve_type == 'likelihood':
            self.message = f'query_liklihood retreival model with smoothing type {self.smooth_type}'
        else:
            raise ValueError(f'Unknown retrieve_type {self.retrieve_type}')

    def retrieve(
            self,
            num_top_results: int | None = None,
            print_details: bool | None = False,
            filename: str | None = None
    ):
        """Retrieve.
        
        Args:
            num_top_results: number of the top results tp return if not None
            print_details: If True, for each query, print when retrival is finish
            filename: filename to save the retrival results. If None, the retrival results will not be saved

        Returns:
            retrieval_results: retrieval resultes for all queries
        """
        
        print(f'start : {self.message}')

        retrieval_results = self._retrieve(num_top_results=num_top_results, print_details=print_details)
        if filename is not None:
            retrieval_results.to_csv(filename,index=False,header=False)

        print('finish : {}. Length is {}'.format(self.message, len(retrieval_results)))

        return retrieval_results
    
    def _retrieve(
            self,
            num_top_results: int | None = None,
            print_details: bool | None = False
    ):
        '''Retrieve according to the query_candidate

        Args:
            num_top_results: number of the top results tp return if not None
            print_details: If True, for each query, print when retrival is finish

        Returns:
            retrieve_all: retrieval resultes for all queries
        '''

        retrieve_all = []
        for qid, values in self.scorer.data_loader.query_candidate.groupby('qid'):
            # calculate the retrieve score for each pid under this qid
            retrieve_qid = [(qid, pid, self._calculate_score(qid, pid)) for pid in values['pid']][:num_top_results]
            retrieve_qid = pd.DataFrame(retrieve_qid, columns=['pid','qid','score']).sort_values(by='score', ascending=False)
            retrieve_all.append(retrieve_qid)
            if print_details:
                print('finish query {}'.format(qid))

        retrieve_all = pd.concat(retrieve_all, ignore_index=True)

        return retrieve_all
    
    def _calculate_score(self, qid:str, pid:str):

        return self.scorer(qid=qid, pid=pid)
    
    def reset_retrieval_method(
            self,
            retrieve_type: Literal['tf-idf', 'bm25', 'likelihood'] | None = None,
            smooth_type: Literal['laplace', 'lidstone', 'dirichlet'] | None = None,
            **kwargs
    ):
        '''Reset the retrival method

        Args:
            retrieve_type: the retrieval type. Can be 'tf-idf', 'bm25' or 'likelihood'.
            smooth_type: the smoothing method to use when retrieve_type is 'likelihood'
                Can be 'laplace', 'lidstone', 'dirichlet'.
            kwargs: the hyperamters for the score function, can be:
                'k1' (float), 'k2' (float), 'b' (float), for 'bm25', Default is 1.2, 100, 0.75 respectively,
                'eps' (float), for 'discounting', Default is 1,
                'mu' (float), for 'dirichlet', Default is 50,
        '''
        flag = True

        if retrieve_type is not None:
            self.retrieve_type = retrieve_type
            flag = False
        if smooth_type is not None:
            self.smooth_type = smooth_type
            flag = False
        if not kwargs:
            for key, item in kwargs.items():
                self.kwargs[key] = item
            flag = False
        
        if flag:
            warn('the retrieval method has not been changed')

        score_type = self._check_determine_score_type()
        self._add_message()
        self.scorer.reset_score_type(score_type, **self.kwargs)

    def _check_determine_score_type(self):
        '''Check the compatibility of retrieve_type, smooth type, and kwargs.
        Also, get the score type according to the retrieve_type and smooth_type.
        
        Returns:
            the score type
        '''
        
        if self.retrieve_type == 'tf-idf' or self.retrieve_type == 'bm25':
            if self.smooth_type is not None:
                warn(f'the retrieval type is {self.retrieve_type}, but the smoothing type is assgined a value {self.smooth_type}')
            return self.retrieve_type
        
        elif self.retrieve_type == 'likelihood':
            if self.smooth_type in ['laplace', 'lidstone', 'dirichlet']:
                return self.smooth_type
            else:
                raise ValueError(f'Unsupported smoothing type {self.smooth_type} for likelihood retrieval method')
        
        else:
            raise ValueError(f'Unsupported retrieval method {self.retrieve_type}')


def calculate_score_tfidf(tf_idf_query: dict, tf_idf_passage: dict) -> float:
    '''Calulate the score for TF-IDF vector space based retrieval model

    Args:
        terms_query: all unique terms in the query
        tf_idf_query_map: a fuction mapping the terms_query to the tf-idf values
        tf_idf_passage_map: a fuction mapping the term within the passage to the tf-idf values.
            return 0 if the passage doesn't have this term

    Returns:
        score: the socre of the query-passage pair for TF-IDF vector space based retrieval model
    '''
    product = 0
    norm1 = 0
    norm2 = 0

    for term, tf_idf_query_term in tf_idf_query.items():
        # only calculate for the common term
        tf_idf_passage_term = tf_idf_passage.get(term, 0)
        if tf_idf_passage_term == 0:
            continue
        product += tf_idf_query_term * tf_idf_passage_term
        norm1 += tf_idf_query_term ** 2
        norm2 += tf_idf_passage_term ** 2

    # query and passage do not have common term
    if product == 0:
        return 0

    score = product / np.sqrt(norm1) / np.sqrt(norm2)

    return(score)


def calculate_score_BM25(
        tf_query: dict, 
        tf_passage: dict, 
        tf_collection: dict,
        passages_length: int,
        ave_length: float,
        num_passages: int,
        k1: float | None = 1.2,
        k2: float | None = 100,
        b: float | None = 0.75
) -> float:
    '''Calculate the score for BM25 model

    Args:
        tf_query: a dict with all terms of the query as keys and tf as values
        tf_passage: a dict with all terms of the passage as keys and tf as values
        tf_collection: a dict with all terms of the collection as keys as tf for the entire collection as values
        ave_length: average length of the passages
        num_passages: number of the passages (i.e. size of the collection)
        k1, k2, b: hyparameters of the score function for BM25. Default is 1.2, 100, 0.75 respectively

    Returns:
        score: the BM25 socre of the query-passage pair
    ''' 
                   
    K = k1 * ((1 - b) + b * passages_length / ave_length)
    
    score = 0
    for term, tf_query_term in tf_query.items():
        tf_passage_term = tf_passage.get(term, None)
        if tf_passage_term is None:
            continue
        tf_collection_term = tf_collection[term]

        score += np.log(
            ((0 + 0.5) / (0 - 0 + 0.5)) / ((tf_collection_term - 0 + 0.5 ) / (num_passages - tf_collection_term - 0 + 0 + 0.5)) \
            * ((k1 + 1) * tf_passage_term) / (K + tf_passage_term) \
            * ((k2 + 1) * tf_query_term) / (k2 + tf_query_term)
        )

    return score


def calculate_score_discounting(
        tf_query: dict, 
        tf_passage: dict,
        passages_length: int,
        num_unique_words: int, 
        eps: float | None = 1
) -> float:
    '''Calculate the log score under Laplace smoothing with or without Lidstone correction for the likelihood model

    Args:
        tf_query: a dict with all terms of the query as keys and tf as values
        tf_passage: a dict with all terms of the passage as keys and tf as values
        passages_length: a dict with pids as keys and length of the passage as values
        num_unique_words: number of unique words in the entire collection
        eps: parammter for the smoothing method. 1 means without Lidstone correction. Default is 1.

    Returns:
        score: the log score
    '''

    score = 0

    for term, tf_query_term in tf_query.items():
        score += tf_query_term * np.log((eps + tf_passage.get(term, 0)) / (eps * num_unique_words + passages_length))

    return score


def calculate_score_laplace(
        tf_query: dict, 
        tf_passage: dict,
        passages_length: int,
        num_unique_words: int
) -> float:
    """Calculate the log score with Laplace smoothing for the likelihood retrieval model
    
    Args:
        tf_query: a dict with all terms of the query as keys and tf as values
        tf_passage: a dict with all terms of the passage as keys and tf as values
        passages_length: a dict with pids as keys and length of the passage as values
        num_unique_words: number of unique words in the entire collection

    Returns:
        score: the log score
    """
    
    return calculate_score_discounting(
        tf_query=tf_query,
        tf_passage=tf_passage,
        passages_length=passages_length,
        num_unique_words=num_unique_words,
        eps=1
    )


def calculate_score_lidstone(
        tf_query: dict, 
        tf_passage: dict,
        passages_length: int,
        num_unique_words: int, 
        eps: float | None = 0.1
) -> float:
    '''Calculate the log score with Laplace smoothing and Lidstone correction for the likelihood retrieval model

    Args:
        tf_query: a dict with all terms of the query as keys and tf as values
        tf_passage: a dict with all terms of the passage as keys and tf as values
        passages_length: a dict with pids as keys and length of the passage as values
        num_unique_words: number of unique words in the entire collection
        eps: parammter for the smoothing method. Default is 0.1

    Returns:
        score: the log score
    '''

    if np.isclose(1, eps):
        warn(f'the eps {eps} is close to 1')

    return calculate_score_discounting(
        tf_query=tf_query,
        tf_passage=tf_passage,
        passages_length=passages_length,
        num_unique_words=num_unique_words,
        eps=eps
    )


def calculate_score_dirichlet(
        tf_query: dict, 
        tf_passage: dict,
        tf_collection: dict,
        passages_length: int,
        length_collection: int,
        mu: float | None = 50
) -> float:
    '''Calculate the log score under dirichlet smoothing for the likelihood model

    Args:
        tf_query: a dict with all terms of the query as keys and tf as values
        tf_passage: a dict with all terms of the passage as keys and tf as values
        tf_collection: a dict with all terms of the collection as keys as tf for the entire collection as values
        passages_length: a dict with pids as keys and length of the passage as values
        length_collection: number of terms in the entire collection
        mu: parammter for the smoothing method. Default is 50.

    Returns:
        score: the log score
    '''
    score = 0
    d = passages_length
    lam = d / (d + mu)

    for term, tf_query_term in tf_query.items():
        tf_passage_term = tf_passage.get(term, None)
        if tf_passage_term is None:
            continue
        score += tf_query_term * np.log(
            lam * tf_passage_term / d + (1-lam) * tf_collection[term] / length_collection
        )
    
    return score