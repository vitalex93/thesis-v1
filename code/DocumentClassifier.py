import spacy
from spacy.matcher import Matcher
from collections import Counter

class DocumentClassifier:
    def __init__(self, pattern_dict):
        self.nlp = spacy.load("en_core_web_sm")
        self.matchers = {}
        for category, patterns in pattern_dict.items():
            matcher = Matcher(self.nlp.vocab)
            #for pattern in patterns:
            matcher.add(category, patterns)
            self.matchers[category] = matcher
    
    def classify(self, text):
        doc = self.nlp(text)
        category_counts = Counter()
        for category, matcher in self.matchers.items():
            matches = matcher(doc)
            count = len(matches)
            category_counts[category] += count
        if not category_counts:
            return None
        most_common_category, _ = category_counts.most_common(1)[0]
        category_ranking = category_counts.most_common()
        return most_common_category, category_ranking
    
    def get_top_n_categories(self, doc_text, n=3):
        doc_categories, ranking = self.classify(doc_text)
        top_n_categories = [cat for cat, rank in ranking[:n] if rank > 0]
        return {doc_text: top_n_categories}
    

