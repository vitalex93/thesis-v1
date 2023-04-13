import textdistance
import string
import nltk

class DocumentSimilarityCalculator:
    
    def __init__(self, algorithm='jaccard', mode='token'):
        self.td = textdistance.algorithms
        self.algorithm = algorithm
        self.mode = mode
        
        if self.mode == 'token':
            self.stopwords = set(nltk.corpus.stopwords.words('english'))
        elif self.mode == 'edit':
            self.punctuation = set(string.punctuation)
        
    def preprocess(self, doc):
        if self.mode == 'token':
            tokens = nltk.word_tokenize(doc.lower())
            tokens = [token for token in tokens if token not in self.stopwords]
            return tokens
        elif self.mode == 'edit':
            doc = ''.join(char for char in doc if char not in self.punctuation)
            return doc
        
    def calculate_similarity(self, doc1, doc2):
        if self.mode == 'token':
            doc1_tokens = self.preprocess(doc1)
            doc2_tokens = self.preprocess(doc2)
            algorithm_dict = {'jaccard': self.td.jaccard.normalized_similarity,
                              'sorensen': self.td.sorensen.normalized_similarity,
                              'tversky': self.td.tversky.normalized_similarity,
                              'overlap': self.td.overlap.normalized_similarity,
                              'tanimoto': self.td.tanimoto.normalized_similarity,
                              'cosine': self.td.cosine.normalized_similarity,
                              'monge_elkan': self.td.monge_elkan.normalized_similarity,
                              'bag': self.td.bag.normalized_similarity}
            similarity_metric = algorithm_dict[self.algorithm]
            #similarity_score = similarity_metric(set(doc1_tokens), set(doc2_tokens))
            similarity_score = similarity_metric(doc1, doc2)
            return similarity_score
        elif self.mode == 'edit':
            doc1_edit = self.preprocess(doc1)
            doc2_edit = self.preprocess(doc2)
            algorithm_dict = {'hamming': self.td.hamming.normalized_similarity,
                              'mlipns': self.td.mlipns.normalized_similarity,
                              'levenshtein': self.td.levenshtein.normalized_similarity,
                              'damerau_levenshtein': self.td.damerau_levenshtein.normalized_similarity,
                              'jaro_winkler': self.td.jaro_winkler.normalized_similarity,
                              'strcmp95': self.td.strcmp95.normalized_similarity,
                              'needleman_wunsch': self.td.needleman_wunsch.normalized_similarity,
                              'gotoh': self.td.gotoh.normalized_similarity,
                              'smith_waterman': self.td.smith_waterman.normalized_similarity}
            similarity_metric = algorithm_dict[self.algorithm]
            similarity_score = similarity_metric(doc1, doc2)
            return similarity_score
        elif self.mode == 'seq':
            doc1_edit = self.preprocess(doc1)
            doc2_edit = self.preprocess(doc2)
            algorithm_dict = {'lcsseq': self.td.lcsseq.normalized_similarity,
                              'lcsstr': self.td.lcsstr.normalized_similarity,
                              'ratcliff_obershelp': self.td.ratcliff_obershelp.normalized_similarity}
            similarity_metric = algorithm_dict[self.algorithm]
            similarity_score = similarity_metric(doc1, doc2)
            return similarity_score

