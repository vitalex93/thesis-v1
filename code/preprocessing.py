import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join words back into a string
    #text = ' '.join(words)
    
    return words


def get_unique_items(list1, list2):
    # Find the intersection of the two lists
    intersection = set(list1) & set(list2)

    # Find the items that are unique to each list
    unique_list1 = [item for item in list1 if item not in intersection]
    unique_list2 = [item for item in list2 if item not in intersection]

    # Return the unique items
    return unique_list1, unique_list2
