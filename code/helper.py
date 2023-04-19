from textmodels import *
from DocumentClassifier import *
from keywords import *
from similarity import *
import pandas as pd
import openpyxl
from collections import Counter
import gensim
import csv


def get_similarity_scores_td(text, documents, algorithm, mode):
    dsc = DocumentSimilarityCalculator(algorithm=algorithm, mode=mode)
    similarity_scores = {}
    for doc in documents:
        similarity_score = dsc.calculate_similarity(text, doc) #cosine_similarity([text], [doc])[0][0]
        similarity_scores[doc] = similarity_score

    # Sort similarity scores in descending order
    similarity_scores = {k: v for k, v in sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)}
    
    return similarity_scores

def get_similarity_scores(text, documents, encoding_method,
                          version, tm, preprocessing=False, fin=False):
    encode_func = tm.encode
    
    # Calculate similarity scores
    similarity_scores = {}
    query_encoding = encode_func(text=text, model=encoding_method, version=version, preprocessing=preprocessing, fin=fin)
    #print(query_encoding)
    for doc in documents:
        doc_encoding = encode_func(text=doc, model=encoding_method, version=version, preprocessing=preprocessing, fin=fin)
        if doc_encoding is not None:
            similarity_score = cosine_similarity([query_encoding], [doc_encoding])[0][0]
            similarity_scores[doc] = similarity_score
        else:
            similarity_scores[doc] = 0
    
    # Sort similarity scores in descending order
    similarity_scores = {k: v for k, v in sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)}
    
    return similarity_scores


def similarities_df(docs, group, encoding_method, cols=['Descriptions'], path='../reports/corpus.xlsx'):
    #model = TextModels(path, cols)
    similarities = {}
    for doc in docs:
        similarities[doc] = get_similarity_scores(doc, group, encoding_method)
    #df = pd.DataFrame.from_dict(similarities, orient='index', columns=group)
    df = pd.DataFrame.from_dict(similarities)
    #df = df.T
    return df


def get_column_values(filename, sheetname, column_letter):
    # Load the workbook
    wb = openpyxl.load_workbook(filename)
    
    # Select the sheet
    sheet = wb[sheetname]
    
    # Get the values in the column
    column = sheet[column_letter]
    values = [cell.value for cell in column]
    values = [value for value in values if value is not None]
    
    # Return the list of values
    return values


def calculate_percentage(first_list, second_list):
    common_items = set(first_list) & set(second_list)
    percentage = len(common_items) / len(first_list) * 100
    return percentage


def get_first_n_keys(dictionary, n):
    keys_list = list(dictionary.keys())[:n]
    return keys_list


def get_similarities_for_values(values, docs, encoding_method, version, tm, preprocessing=False, n=10, fin=False):
    reports = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9']
    similarities_dict = {}
    for i in range(len(reports)):
        similarities_dict[reports[i]] = get_first_n_keys(get_similarity_scores(text=values[i], documents=docs, 
                                                                          encoding_method=encoding_method, 
                                                                          version=version, tm=tm, preprocessing=preprocessing, fin=fin), n)
    return similarities_dict


def save_df_to_excel(dataframe, file_name, sheet_name):
    # create an Excel writer object
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

    # write the dataframe to a specific sheet
    dataframe.to_excel(writer, sheet_name=sheet_name)

    # save the file
    writer.save()


def match_lists(lst, dct):
    """
    Matches each item in `lst` with the list in `dct` that has the most common items.
    Returns a dictionary with the item of `lst` as key and the keys of `dct` as values.
    """
    # Create a dictionary to store the counts of each item in the lists in `dct`
    counts = {}
    for k, v in dct.items():
        counts[k] = Counter(v)
    
    # Find the list in `dct` with the most common items for each item in `lst`
    matches = {}
    for item in lst:
        max_count = 0
        max_list = None
        for k, v in counts.items():
            count = v.get(item, 0)
            if count > max_count:
                max_count = count
                max_list = k
        matches[item] = max_list
    
    # Return a dictionary with the keys of `dct` as values
    result = {}
    for k in dct.keys():
        result[k] = [item for item, lst in matches.items() if lst == k]
    
    return result


def df_to_lists(df):
    """
    Convert all columns of a Pandas DataFrame to lists.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to convert to lists.

    Returns:
    --------
    list
        A list of lists, where each inner list corresponds to one column of the DataFrame.
    """
    # Initialize an empty list to store the result
    result = []

    # Loop over the columns of the DataFrame and convert each one to a list
    for col in df.columns:
        col_list = df[col].tolist()
        result.append(col_list)

    return result


def results_to_targets(descriptions, targets, model, version, tm, preprocessing=False, n=10, path='../reports/corpus.xlsx', i=0, fin=False):
    
    results_dict = get_similarities_for_values(values=descriptions,docs=targets,encoding_method=model,
                                               version=version,tm=tm,preprocessing=preprocessing,n=n, fin=fin)
    results_df = pd.DataFrame.from_dict(results_dict)
    results = df_to_lists(results_df)
    if preprocessing == False:
        ground_truth = {'R1':get_column_values(path, 'R1', 'C'),
                        'R2':get_column_values(path, 'R2', 'C'),
                        'R3':get_column_values(path, 'R3', 'C'),
                        'R4':get_column_values(path, 'R4', 'C'),
                        'R5':get_column_values(path, 'R5', 'C'),
                        'R6':get_column_values(path, 'R6', 'C'),
                        'R7':get_column_values(path, 'R7', 'C'),
                        'R8':get_column_values(path, 'R8', 'C'),
                        'R9':get_column_values(path, 'R9', 'C')}
    elif preprocessing == True:
        ground_truth = {'R1':[preprocess_text(element) for element in get_column_values(path, 'R1', 'C')],
                        'R2':[preprocess_text(element) for element in get_column_values(path, 'R2', 'C')],
                        'R3':[preprocess_text(element) for element in get_column_values(path, 'R3', 'C')],
                        'R4':[preprocess_text(element) for element in get_column_values(path, 'R4', 'C')],
                        'R5':[preprocess_text(element) for element in get_column_values(path, 'R5', 'C')],
                        'R6':[preprocess_text(element) for element in get_column_values(path, 'R6', 'C')],
                        'R7':[preprocess_text(element) for element in get_column_values(path, 'R7', 'C')],
                        'R8':[preprocess_text(element) for element in get_column_values(path, 'R8', 'C')],
                        'R9':[preprocess_text(element) for element in get_column_values(path, 'R9', 'C')]}        
    matched = common_items(results[i], ground_truth)
    return matched


def common_items(input_list, input_dict):

    """
    Returns a dictionary with the common items between the input list and each value list in the input dictionary.

    Parameters:
    -----------
    input_list : list
        The input list to compare with the values of the input dictionary.
    input_dict : dict
        The input dictionary with lists as values to compare with the input list.

    Returns:
    --------
    dict
        A dictionary with the keys of the input dictionary as keys and the common items as values.
    """
    # Initialize an empty dictionary to store the result
    result = {}

    # Loop over the items in the input dictionary
    for key, value in input_dict.items():
        # Find the common items between the input list and the current value list
        common = set(input_list).intersection(set(value))
        # Add the common items to the result dictionary with the current key as key
        result[key] = list(common)

    return result


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
    text = ' '.join(words)
    
    return text


def candidate_templates(descriptions, targets, model, version, tm, path, preprocessing=False, n=10, fin=False):
    print(f'==================== {model} ====================')
    for i in range(9):
        d = results_to_targets(descriptions=descriptions, targets=targets, model=model, 
                               version=version, tm=tm, preprocessing=preprocessing, n=n, path=path, i=i, fin=fin)
        print(f'Candidate templates for Q{i+1}')
        print(d)

def get_unique_items(list1, list2):
    # Find the intersection of the two lists
    intersection = set(list1) & set(list2)

    # Find the items that are unique to each list
    unique_list1 = [item for item in list1 if item not in intersection]
    unique_list2 = [item for item in list2 if item not in intersection]

    # Return the unique items
    return unique_list1, unique_list2


def percentage_of_words_in_word2vec_vocabulary(word2vec_model, sentence_list):
    # Load the pretrained Word2Vec model
    model = api.load(word2vec_model)

    # Create an empty dictionary to store the percentages
    percentage_dict = {}

    # Iterate through each sentence in the list
    for sentence in sentence_list:
        # Split the sentence into a list of words
        words_list = sentence.split()

        # Count the number of words in the sentence
        total_words = len(words_list)

        # Count the number of words that exist in the model's vocabulary
        words_found = 0
        for word in words_list:
            if word in model:
                words_found += 1

        # Calculate the percentage of words found
        percentage_found = (words_found / total_words) * 100

        # Add the sentence and percentage to the dictionary
        percentage_dict[sentence] = f'{percentage_found:.2f}%'

    return percentage_dict


'''def evaluation_metrics(results, metric, encoding_method,
                        version_bow=None,version_tfidf=None, version_w2v=None, version_sbert=None, k=10):
    if encoding_method == 'bow':
        model = joblib.load(version_bow)
    elif encoding_method == 'tfidf':
        model = joblib.load(version_tfidf)
    elif encoding_method == 'word2vec':
        model = api.load(version_w2v)
    elif encoding_method == 'sbert':
        model =  SentenceTransformer(version_sbert)

    if metric == 'precision':
        result = {}
        for key, value in results.items():
            n = len(value)
            percentage = (n / k) * 100
            result[key] = percentage
        return result
    elif metric == 'recall':
        pass'''

def calculate_percentage(dict1, dict2, key1, key2, metric):
    if key1 not in dict1 or key2 not in dict2:
        return "Invalid keys provided"
    
    list1 = dict1[key1]
    list2 = dict2[key2]
    
    if not list1 or not list2:
        return "One or both of the lists is empty"
    
    count = 0
    for element in list1:
        if element in list2:
            count += 1
    if metric == 'precision':
        percentage = (count / len(list1)) * 100
    elif metric == 'recall':
        percentage = (count / len(list2)) * 100
    elif metric == 'r_precision':
        count = 0
        for element in list1[:len(list2)]:
            if element in list2:
                count += 1
        percentage = (count / len(list2)) * 100
    
    return percentage


'''def precision(values, docs, encoding_method,
            version, tm, preprocessing, path, n):
    ground_truth = {'R1':get_column_values(path, 'R1', 'C'),
                'R2':get_column_values(path, 'R2', 'C'),
                'R3':get_column_values(path, 'R3', 'C'),
                'R4':get_column_values(path, 'R4', 'C'),
                'R5':get_column_values(path, 'R5', 'C'),
                'R6':get_column_values(path, 'R6', 'C'),
                'R7':get_column_values(path, 'R7', 'C'),
                'R8':get_column_values(path, 'R8', 'C'),
                'R9':get_column_values(path, 'R9', 'C')}
    results = get_similarities_for_values(values=values, docs=docs, encoding_method=encoding_method,
                                      version=version, tm=tm, preprocessing=preprocessing, n=n)
    precision = {}
    for i in range(9):
        query = 'Q' + str(i+1)
        report = 'R' + str(i+1)
        precision[query] = calculate_percentage(results, ground_truth,query,report)
    return precision'''


def evaluation_metrics(metric, values, docs, encoding_method, version, tm, preprocessing, path, n, fin=False):
    ground_truth = {'R'+str(i+1):get_column_values(path, 'R'+str(i+1), 'C') for i in range(9)}
    metrics = {}
    for i in range(9):
        query = 'Q' + str(i+1)
        report = 'R' + str(i+1)
        metrics[query] = calculate_percentage(get_similarities_for_values(values=values, docs=docs, encoding_method=encoding_method,
                                      version=version, tm=tm, preprocessing=preprocessing, n=n, fin=fin), ground_truth,query,report, metric)
    return metrics


'''def dict_to_dataframe(input_dict):
    # create an empty DataFrame with the same columns as the internal dictionaries
    columns = list(input_dict[next(iter(input_dict))].keys())
    df = pd.DataFrame(columns=columns)

    # loop through the external dictionary and add each row to the DataFrame
    for key, values_dict in input_dict.items():
        row = [values_dict[column] for column in columns]
        df.loc[key] = row

    return df
    input_dict = {
    "dict1": {"a": 1, "b": 2, "c": 3},
    "dict2": {"a": 4, "b": 5, "c": 6},
    "dict3": {"a": 7, "b": 8, "c": 9}
}

df = dict_to_dataframe(input_dict)
print(df)
'''


def dict_to_dataframe(input_dict):
    # Create an empty DataFrame with the column names as the keys of the input_dict
    df = pd.DataFrame(columns=input_dict.keys())
    
    # Loop through each key-value pair in the input_dict
    for outer_key in input_dict.keys():
        # Add a new row to the DataFrame for each key-value pair in the inner_dict
        for inner_key, inner_value in input_dict[outer_key].items():
            df.loc[inner_key, outer_key] = inner_value
    
    return df

def rule_based_candidates(patterns, descriptions, n, mode, path, sheet, col=['A','B','C','D']):
    classifier = DocumentClassifier(patterns)
    candidates_dict = {}
    if mode == 'columns':
        for description in descriptions:
            #if mode == 'columns':
            candidates_dict_single = classifier.get_top_n_categories(doc_text=description, n=n)
            candidates_dict.update(candidates_dict_single)
    elif mode == 'rows':
        rg_list = [['RowGroup1'], ['RowGroup2'],['RowGroup3'], ['RowGroup4']]
        n = 1
        for description in descriptions:
            candidates_dict_single = classifier.get_top_n_categories(doc_text=description, n=n)
            candidates_dict.update(candidates_dict_single)
        for i in range(len(rg_list)):
            for description in descriptions:
                if candidates_dict[description] == rg_list[i]:
                    candidates_dict[description] = get_column_values(path, sheet, col[i])
        for key in candidates_dict.keys():
            candidates_dict[key] = [[val] for val in candidates_dict[key]]
    return candidates_dict

def create_csv(list1, list2, filename):
    # create a new CSV file and write the headers
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list1)
        
        # write each row of values from list2, replacing missing values with blank strings
        for row in list2:
            new_row = [val if val else "" for val in row]  # replace missing values with blank strings
            if len(new_row) < len(list1):  # if the row is shorter than the header, add blank values to the end
                diff = len(list1) - len(new_row)
                new_row.extend(["" for i in range(diff)])
            writer.writerow(new_row)

def rule_based_templates(patterns_measures, patterns_rows, descriptions, n, path, sheet):
    measures = rule_based_candidates(patterns=patterns_measures, descriptions=descriptions,
                                     n=n, mode='columns', path=path, sheet=sheet)
    rows = rule_based_candidates(patterns=patterns_rows, descriptions=descriptions,
                                 n=n, mode='rows', path=path, sheet=sheet)
    for i in range(len(descriptions)):
        measure_list = measures[descriptions[i]]
        rows_list = rows[descriptions[i]]
        filename = '../results/templates/rule_based/R' + str(i+1) + '_rule_based_template.csv'
        create_csv(measure_list, rows_list, filename=filename)

def kw_extraction(library, text, max_kw, stopwords, max_length=3, min_length=1, language='en', dedupLim=0.9):
    kw_extractor = KeywordExtractor(language=language)
    if library == 'yake':
        yake_kw = kw_extractor.extract_yake(text=text, top=max_kw, 
                                             dedupLim=dedupLim, stopwords=stopwords)
        yake_kw_sentence = ' '.join(yake_kw)
        return yake_kw, yake_kw_sentence
    elif library == 'rake':
        rake_kw = kw_extractor.extract_rake(text=text, stopwords=stopwords, max_kw=max_kw,
                                             max_length=max_length, min_length=min_length)
        rake_kw_sentence = ' '.join(rake_kw)
        return rake_kw, rake_kw_sentence
    
def build_kw_corpus(mode, descriptions, targets):
    kwl = []
    for i in range(len(descriptions)):
        kwe = KeywordExtractor()
        kws = kwe(mode=mode, text=descriptions[i])
        kwl.append(kws)
    kwl = kwl + targets
    return kwl




        

            
   













