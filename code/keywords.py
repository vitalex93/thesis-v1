import yake
from rake_nltk import Rake
#from gensim.summarization import keywords
#from gensim.summarization import textrank

class KeywordExtractor:
    def __init__(self, language='en', max_keywords=10):
        self.language = language
        #self.max_keywords = max_keywords

    def extract_yake(self, text, stopwords, top, n=3, dedupLim=0.9):
        kw_extractor = yake.KeywordExtractor(lan=self.language, top=top, n=n,
                                             dedupLim=dedupLim, stopwords=stopwords)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]

    def extract_rake(self, text, max_kw, stopwords=None, max_length=3, min_length=2, ranking_metric='word_frequency'):
        r = Rake(stopwords=stopwords, max_length=max_length, min_length=min_length, ranking_metric=ranking_metric )
        r.extract_keywords_from_text(text)
        return r.get_ranked_phrases()[:max_kw]

    def extract_gensim_keywords(self, text):
        return keywords(text, lemmatize=True, words=self.max_keywords, split=True)

    def extract_textrank(self, text):
        return textrank(text, words=self.max_keywords, lemmatize=True, split=True)

