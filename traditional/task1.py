import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from typing import Tuple

def extract_frequencies(filename:str, remove = False) -> Tuple[int,dict]:
    '''extract frequencies of terms (1-grams) from .txt file

    Input:
        filename: path of the .txt file
        remove: bool, True indicates removing stop words
    Output:
        size: size of the vocabulary
        frequencies: a dict with tokens (terms) as keys and frequencies as values
    '''

    with open(filename,'r',encoding='utf-8') as f:
        content = f.read()

    content = re.sub(r'[^a-zA-Z\s]',' ',content).lower()
    tokens = content.split()

    frequencies = {}
    for token in tokens:
        if token in frequencies.keys():
            frequencies[token] += 1
        else:
            frequencies[token] = 1

    size = len(frequencies)

    if remove:
        nltk.download('stopwords')
        stopwords_eng = stopwords.words('english')
        for word in stopwords_eng:
            frequencies.pop(word, None)

    return size, frequencies
    
def norm_fre_zipfian(fre, s=1):
    '''get the normalized frequencies and the zipfian value as well as their difference

    Input:
        fre: a dict representing the (unnormalized) frequencies of the terms 
        s: parameter of zipfian distribution
    Output:
        (fre_norm, zipfian): a tuple (normalized frequencies, zipfian value)
    '''
    fre_sorted = sorted(fre.items(),key = lambda item:item[1],reverse= True)
    fre_norm = np.array([v for _,v in fre_sorted])
    fre_norm = fre_norm / np.sum(fre_norm)

    n = len(fre_norm)
    zipfian = np.arange(1,n+1,dtype='float') ** (-s) / np.sum(np.arange(1,n+1,dtype='float') ** (-s))

    diff = np.average((fre_norm -zipfian)**2)
    
    return fre_norm,zipfian,diff

def plot(fre_norm,zipfian,filename1 = None, filename2 = None, string_titile = ''):
    '''plot the fre_norm and zipfian both in norm and log-log

    Input:
        fre_norm: a vector of normalized frequencied ordered desc.
        zipfian: a vector of zipfian values ordered desc
        filename1: the filenmame of plot
        filenmae2: the filenames of log-log plot
        string_titile: string added to the filenames2 for results after removing stop words
    Output:
        (not return): two figures - normal and log-log, showing the distrubution and zipfian
    '''

    x = np.arange(1,len(fre_norm)+1)

    if filename1 is not None:
        _, ax = plt.subplots(figsize = (6, 6))
        ax.plot(x,fre_norm,'b-',markersize=5,markerfacecolor='none', label = 'empirical')
        ax.plot(x,zipfian,'r--',markersize=5,markerfacecolor='none', label = "Zipf's law")
        ax.set_xlabel(r'term frequency rank $k$')
        ax.set_ylabel(r'normalized frequency $f$')
        ax.set_title('distribution comparison')

        plt.legend()
        plt.savefig(filename1)
    
    if filename2 is not None:
        _, ax = plt.subplots(figsize = (6, 6))
        ax.loglog(x,fre_norm,'b-',markersize=5,markerfacecolor='none', label = 'empirical')
        ax.loglog(x,zipfian,'r--',markersize=5,markerfacecolor='none', label = "Zipf's law")
        ax.set_xlabel(r'term frequency rank $k$')
        ax.set_ylabel(r'normalized frequency $f$')
        ax.set_title('log-log distribution comparison' + string_titile)

        plt.legend()
        plt.savefig(filename2)


if __name__ == '__main__':
    # before removing stop words
    size, frequencies = extract_frequencies('passage-collection.txt')
    print('size of the vocabulary : {}'.format(size))
    fre, zipfian, diff = norm_fre_zipfian(frequencies)
    print('difference with stop wprds : {}'.format(diff))
    plot(fre,zipfian,'task1.png','task1-loglog.png')

    # after removing stop words
    # to speed up thus not using extract_frequencies
    nltk.download('stopwords')
    stopwords_eng = stopwords.words('english')
    for word in stopwords_eng:
        frequencies.pop(word, None)
    fre, zipfian, diff = norm_fre_zipfian(frequencies)
    print('difference without stop wprds : {}'.format(diff))
    plot(fre,zipfian, filename2='task1-loglog2.png',string_titile=' without stop words')

