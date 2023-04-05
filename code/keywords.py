import yake
from rake_nltk import Rake
from gensim.summarization import keywords
from gensim.summarization import textrank

class KeywordExtractor:
    def __init__(self, language='en', max_keywords=10):
        self.language = language
        self.max_keywords = max_keywords

    def extract_yake(self, text):
        kw_extractor = yake.KeywordExtractor(lan=self.language, top=self.max_keywords)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]

    def extract_rake(self, text):
        r = Rake(language=self.language, max_length=3)
        r.extract_keywords_from_text(text)
        return r.get_ranked_phrases()[:self.max_keywords]

    def extract_gensim_keywords(self, text):
        return keywords(text, lemmatize=True, words=self.max_keywords, split=True)

    def extract_textrank(self, text):
        return textrank(text, words=self.max_keywords, lemmatize=True, split=True)

