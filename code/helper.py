from textmodels import *
import pandas as pd
import openpyxl
from collections import Counter

def get_similarity_scores(text, documents, encoding_method,cols=['Descriptions'],path='../reports/corpus.xlsx'):
    # Initialize TextModels object
    tm = TextModels(excel_path=path, columns=cols)
    
    # Build encoding model based on encoding_method argument
    if encoding_method == 'bow':
        tm.build_bow_model()
        encode_func = tm.encode_bow
    elif encoding_method == 'tfidf':
        tm.build_tfidf_model()
        encode_func = tm.encode_tfidf
    elif encoding_method == 'word2vec':
        #tm.build_word2vec_model()
        encode_func = tm.encode_word2vec
    elif encoding_method == 'sbert':
        encode_func = tm.encode_sentence_bert
    elif encoding_method == 'cbow':
        tm.build_cbow_model()
        encode_func = tm.encode_cbow
    else:
        print("Invalid encoding method")
        return None
    
    # Calculate similarity scores
    similarity_scores = {}
    query_encoding = encode_func(text)
    for doc in documents:
        doc_encoding = encode_func(doc)
        similarity_score = cosine_similarity([query_encoding], [doc_encoding])[0][0]
        similarity_scores[doc] = similarity_score
    
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


def get_similarities_for_values(values, docs, encoding_method, n):
    similarities_dict = {}
    for value in values:
        similarities_dict[value] = get_first_n_keys(get_similarity_scores(value, docs, encoding_method), n)
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


def results_to_targets(descriptions, targets, model, n, path='../reports/corpus.xlsx', i=0):
    
    results_dict = get_similarities_for_values(descriptions,targets,model,n)
    results_df = pd.DataFrame.from_dict(results_dict)

    results = df_to_lists(results_df)
    ground_truth = {'R1':get_column_values(path, 'R1', 'C'),
                    'R2':get_column_values(path, 'R2', 'C'),
                    'R3':get_column_values(path, 'R3', 'C'),
                    'R4':get_column_values(path, 'R4', 'C'),
                    'R5':get_column_values(path, 'R5', 'C'),
                    'R6':get_column_values(path, 'R6', 'C'),
                    'R7':get_column_values(path, 'R7', 'C'),
                    'R8':get_column_values(path, 'R8', 'C'),
                    'R9':get_column_values(path, 'R9', 'C')}

    matched = match_lists(results[i], ground_truth)

    return matched













