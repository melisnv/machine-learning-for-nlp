import csv
import pandas as pd
import spacy
from collections import Counter
import en_core_web_sm
nlp = spacy.load("en_core_web_sm")

nlp.max_length = 1699664 #1030000


def read_in_conll_file(path, delimiter='\t'):
    all_sentences = list()

    with open(path) as file:
        infile = csv.reader(file, delimiter='\t', quotechar='|')
        for row in infile:
            all_sentences.append(row)

    return all_sentences



def data_arrangement(file_path):

    sentences = read_in_conll_file(file_path)
    df = pd.DataFrame(sentences)
    data = df[[0, 3]]
    data = data.rename(columns={0: "word", 3: "ner"})
    data = data.dropna(axis=0, how='all')  # remove None

    return data


def length_of_data(data):

    return len(data)

def distribution_of_data(data):

    return data["ner"].value_counts()


def sample_data(data):
    sample_data = data[:50000]

    return sample_data

def tokenize_data(data):
    tokens = nlp(''.join(str(data.word.tolist())))

    return tokens


def common_features(tokens):
    items = [x.text for x in tokens.ents]

    return Counter(items).most_common(30)


def extract_tags(token,label:str):
    taglist = []
    for ent in token.ents:
        if ent.label_ == label:
            taglist.append(ent.text)

    tag_counts = Counter(taglist).most_common(20)
    df_tag = pd.DataFrame(tag_counts, columns=['text', 'count'])
    df_tag.to_csv('distribution_tag.csv', index=False)

    return df_tag


path_train = "../data/conll2003.train.conll"
path_dev = "../data/conll2003.dev.conll"
train_sentences = read_in_conll_file(path_train)
train_data = data_arrangement(path_train)
dist_of_data = distribution_of_data(train_data)
sample_data =sample_data(train_data)
tokens = tokenize_data(sample_data)
data_features = common_features(tokens)
tag_distribution = extract_tags(tokens,"LOC")
print(tag_distribution)