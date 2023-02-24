import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import gensim.downloader as api
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np


class TextModels:

    def __init__(self, excel_path, columns,word2vec_version="word2vec-google-news-300",w2v_window=5, w2v_workers=4,
                 sbert_model="bert-base-nli-mean-tokens", cbow_window=5):
        self.df = pd.read_excel(excel_path)
        self.columns = columns
        self.bow_model = None
        self.tfidf_model = None
        self.w2v_model = None
        self.sbert_model = SentenceTransformer(sbert_model)
        self.cbow_model = None
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        self.word2vec_model = api.load(word2vec_version)
        self.nlp = spacy.load('en_core_web_sm')
        self.w2v_window = w2v_window
        self.w2v_workers = w2v_workers
        self.cbow_window = cbow_window

    def build_bow_model(self):
        # Build a bag-of-words (BOW) model from the input columns
        self.bow_model = CountVectorizer()
        texts = []
        for col in self.columns:
            texts += self.df[col].tolist()
        self.bow_model.fit_transform(texts)
        self.bow_vectorizer = self.bow_model
        #OK

    def build_tfidf_model(self):
        # Build a TF-IDF model from the input columns
        self.tfidf_model = TfidfVectorizer()
        texts = []
        for col in self.columns:
            texts += self.df[col].tolist()
        self.tfidf_model.fit_transform(texts)
        self.tfidf_vectorizer = self.tfidf_model
        #OK

    '''def build_w2v_model(self):
        # Build a Word2Vec model from the input columns
        texts = []
        for col in self.columns:
            texts += self.df[col].tolist()
        sentences = [self.nlp(text.lower()) for text in texts]
        sentences = [[word.text for word in sentence] for sentence in sentences]
        self.w2v_model = Word2Vec(sentences, window=self.w2v_window, workers=self.w2v_workers)


    def build_cbow_model(self):
        # Build a CBOW model from the input columns
        texts = []
        for col in self.columns:
            texts += self.df[col].tolist()
        sentences = [self.nlp(text.lower()) for text in texts]
        sentences = [[word.text for word in sentence] for sentence in sentences]
        self.cbow_model = Word2Vec(sentences, window=self.cbow_window, workers=self.w2v_workers, sg=0)'''

    def encode_bow(self, text):
        # Encode a document using the BOW model
        doc_bow = self.bow_vectorizer.transform([text])
        return doc_bow.toarray()[0]
        #OK

    def encode_tfidf(self, text):
        # Encode a document using the TF-IDF model
        doc_tfidf = self.tfidf_vectorizer.transform([text])
        return doc_tfidf.toarray()[0]
        #OK

    def encode_word2vec(self, sentence):
        # Encode a document using the TF-IDF model
        words = sentence.split()
        embeddings = [self.word2vec_model[word] for word in words if word in self.word2vec_model]
        sentence_embedding = sum(embeddings) / len(embeddings)
        return sentence_embedding
        #OK

    def encode_sentence_bert(self, sentence):
        # Encode a document using pretrained sentence Bert model
        sentence_embedding = self.sbert_model.encode(sentence, convert_to_numpy =True)
        return sentence_embedding
        #OK

    
    def encode_cbow(self, text):
        # Encode a document using the CBOW model
        if self.cbow_model is not None:
            text = [word for word in text.split() if word in self.cbow_model.wv.vocab]
            doc_cbow = sum(self.cbow_model[text])/len(text)
        else:
            print("Error: CBOW model not loaded")
            doc_cbow = None
        return doc_cbow

    def similarities(self, doc1, doc2, model='tfidf'):
        """
        Computes the similarity score between two documents using the specified model.

        Parameters:
        doc1 (str): The first document to compare.
        doc2 (str): The second document to compare.
        model (str): The name of the model to use for similarity calculation. Valid options are 'bow', 'tfidf', 
                     'word2vec', 'sbert', and 'cbow'. Default is 'tfidf'.

        Returns:
        float: The similarity score between the two documents.
        """
        if model == 'bow':
            vec1 = self.encode_bow(doc1)
            vec2 = self.encode_bow(doc2)
            score = cosine_similarity([vec1], [vec2])[0][0]
        elif model == 'tfidf':
            vec1 = self.encode_tfidf(doc1)
            vec2 = self.encode_tfidf(doc2)
            score = cosine_similarity([vec1], [vec2])[0][0]
        elif model == 'word2vec':
            vec1 = self.encode_word2vec(doc1)
            vec2 = self.encode_word2vec(doc2)
            score = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
        elif model == 'sbert':
            vec1 = self.encode_sentence_bert(doc1)
            vec2 = self.encode_sentence_bert(doc2)
            score = cosine_similarity([vec1], [vec2])[0][0]
        elif model == 'cbow':
            vec1 = self.encode_cbow(doc1)
            vec2 = self.encode_cbow(doc2)
            if vec1 is not None and vec2 is not None:
                score = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
            else:
                score = 0.0
        else:
            print("Error: Invalid model name")
            score = 0.0

        return score




        '''['fasttext-wiki-news-subwords-300',
 'conceptnet-numberbatch-17-06-300',
 'word2vec-ruscorpora-300',
 'word2vec-google-news-300',
 'glove-wiki-gigaword-50',
 'glove-wiki-gigaword-100',
 'glove-wiki-gigaword-200',
 'glove-wiki-gigaword-300',
 'glove-twitter-25',
 'glove-twitter-50',
 'glove-twitter-100',
 'glove-twitter-200',
 '__testing_word2vec-matrix-synopsis']'''

