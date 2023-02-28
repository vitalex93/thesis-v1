from textmodels import *
import pandas as pd
import openpyxl

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









