import re
import task1

def invert_index(filename:str,vocabulary:set):
    '''generate inverted indices for terms in the vocabulary

    Input: 
        filename: path of the query-passage file
        vocabulary: a set containing the terms in the vocabulary
        
    Output:
        inverted_indices: a dict whose keys are all terms in the vobavulary,\
            values are dicts with pids as keys and frequencies as values
        len(pid) : the number of passages (the size of the collection)
        query_candidate: a dict with queries as keys and the lists of the pids as values
    '''

    print('start : calculate inverted indices')
    pattern = re.compile(r'[^a-zA-Z\s]')
    pids = set()
    inverted_indices = {k:{} for k in vocabulary}
    query_candidate = {}

    with open(filename,'r',encoding='utf-8') as f:
        for item in f:
            details = item.split('\t')
            qid = details[0]
            pid = details[1]
            passage = details[3]

            # get the candidate pids for each query
            if qid not in query_candidate.keys():
                query_candidate[qid] = [pid]
            else:
                query_candidate[qid].append(pid)

            # calculate inverted indices
            if pid in pids:
                continue
            pids.add(pid)
            tokens = pattern.sub(' ',passage).lower().split()

            for token in tokens:
                if token not in vocabulary:
                    continue
                if pid not in inverted_indices[token]:
                    inverted_indices[token][pid] = 1
                else:
                    inverted_indices[token][pid] += 1

    print('finish : calculate inverted indices')

    return inverted_indices, len(pids), query_candidate

if __name__ == '__main__':

    length, fre = task1.extract_frequencies('passage-collection.txt',True)
    inverted_indices, size, query_candidate = invert_index('candidate-passages-top1000.tsv', set(fre.keys()))