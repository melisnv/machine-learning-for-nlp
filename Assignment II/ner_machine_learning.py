import gensim as gensim
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import csv
from gensim.models import KeyedVectors
from sys import argv


def extract_embeddings_as_features_and_gold(conllfile,word_embedding_model):
    '''
    Function that extracts features and gold labels using word embeddings
    
    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    ### This code was partially inspired by code included in the HLT course, obtained from https://github.com/cltl/ma-hlt-labs/, accessed in May 2020.
    labels = []
    features = []
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    for row in csvreader:
        #check for cases where empty lines mark sentence boundaries (which some conll files do).
        if len(row) > 3:
            if row[0] in word_embedding_model:
                vector = word_embedding_model[row[0]]
            else:
                vector = [0]*300
            features.append(vector)
            labels.append(row[-1])
    return features, labels


def extract_features_and_labels(trainingfile):
    
    data = []
    targets = []
    # TIP: recall that you can find information on how to integrate features here:
    # https://scikit-learn.org/stable/modules/feature_extraction.html
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                feature_dict = {'token':token}
                data.append(feature_dict)
                #gold is in the last column
                targets.append(components[-1])
    return data, targets
    
def extract_features(inputfile):
   
    data = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                feature_dict = {'token':token}
                data.append(feature_dict)
    return data



def extract_features_and_selected_labels(trainingfile, selected_features):
    '''
    Extract features and gold labels from a preprocessed file with the training data and return them as lists
    
    :param trainingfile: path to training file
    :param selected_features: list of features that will be used to train the model
    
    :type trainingfile: string
    :type selected_features: list
    
    :return features: features as a list of dictionaries
    :return gold_labels: list of gold labels
    '''
    features = []
    gold_labels = []
    
    conllinput = open(trainingfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    
    for row in csvreader:
        feature_value = {}
        # Only extract the selected features
        for feature_name in selected_features:
            row_index = feature_indexes.get(feature_name)
            feature_value[feature_name] = row[row_index]
        features.append(feature_value)
        
        # Gold is in the third column
        gold_labels.append(row[3])
                
    return features, gold_labels

def create_classifier(train_features, train_targets, modelname):
   
    if modelname ==  'logreg':
        # TIP: you may need to solve this: https://stackoverflow.com/questions/61814494/what-is-this-warning-convergencewarning-lbfgs-failed-to-converge-status-1
        model = LogisticRegression()
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
    model.fit(features_vectorized, train_targets)
    
    return model, vec
    
    
def classify_data(model, vec, inputdata, outputfile):
  
    features = extract_features(inputdata)
    features = vec.transform(features)
    predictions = model.predict(features)
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()

def classify_data_embeddings(model, inputdata, outputfile, word_embedding_model):
    '''
    This function creates a classifier for making predictions embedded data
    
    input model: classifier that will make predictions
    input inputdata: path to input data
    input outputfile: path to output file, where the predictions for each feature will be written
    input word_embedding_model : embedding model
    '''
    # extracting features
    features = extract_embeddings_features(inputdata,word_embedding_model)
    
    # making predictions with extracted features
    predictions = model.predict(features)
    
    # Write results to an outputfile
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()

def create_classifier_embeddings(train_features, train_labels):
    '''
    Create an SVM classifier and train it with vectorized features and corresponding gold labels
    
    input train_features: features to be transformed into vectors
    input train_labels: gold labels corresponding to features
    
    output model: trained classifier
    '''

    model = LinearSVC(max_iter=10000)
    model.fit(train_features, train_labels)
    
    return model


def main(system_type, argv=None):
    
    #a very basic way for picking up commandline arguments
    if argv is None:
        argv = sys.argv    
    
    trainingfile_f = argv[1] # "datas/conll2003.train.conll_extracted_features.conll"
    inputfile_f = argv[2] #"datas/conll2003.dev.conll_extracted_features.conll"
    outputfile_f = argv[3] #"output.conll2003_features"
    trainingfile =argv[4]
    inputfile = argv[5]
    outputfile = argv[6]
    language_model = argv[7] #"datas/GoogleNews-vectors-negative300.bin"

    
    if system_type == "with_features":
    # selecting features to train the model
        selected_features = ["token","pos","tag","previous","latter","capitals","stemm","lemma"]

        # getting the selected training features and gold labels
        training_features, gold_labels = extract_features_and_selected_labels(trainingfile_f,selected_features)

        # Training three different models with the features, 
        # Classifying the data and writing the result to new conll files
        for modelname in ['logreg', 'NB', 'SVM']:

            ml_model, vec = create_classifier(training_features, gold_labels, modelname)
            classify_data(ml_model, vec, inputfile_f, outputfile_f.replace('.conll','.' + modelname + '.conll'),selected_features)

            dataframe = pd.read_table(outputfile_f.replace('.conll','.' + modelname + '.conll'))
            dataframe = dataframe.set_axis([*dataframe.columns[:-1], 'NER2'], axis=1, inplace=False)
            dataframe.to_csv(outputfile.replace('.conll','.' + modelname + '.conll'), sep='\t')
        
    elif system_type == "word_embeddings":

    # creating a language model
        language_model = gensim.models.KeyedVectors.load_word2vec_format(language_model, binary=True)

        # extracting the features and gold label
        training_features, gold_labels = extract_embeddings_as_features_and_gold(trainingfile, language_model)

        # creating the classification model
        ml_model = create_classifier_embeddings(training_features[:1], gold_labels[:1])
        classify_data_embeddings(ml_model, inputfile, outputfile.replace('.conll','.embedded.conll'), language_model)

        data_frame = pd.read_table(outputfile.replace('.conll','.embedded.conll'))
        data_frame = data_frame.set_axis([*data_frame.columns[:-1], 'NER2'], axis=1, inplace=False)
        data_frame.to_csv(outputfile.replace('.conll','.embedded2.conll'), sep='\t')
    


if __name__ == '__main__':
    import sys
    print('please input path to training file with extracted features as argv[1] and path to dev file as argv[2] path to output file as argv[3] path to standard traingfile as argv[4] path to standard devfile as argv[5] and outputfile name for SVM with embeddings as argv[6] and finally path to language model as argv[7]')
    trainingfile_f = argv[1] # "datas/conll2003.train.conll_extracted_features.conll"
    inputfile_f = argv[2] #"datas/conll2003.dev.conll_extracted_features.conll"
    outputfile_f = argv[3] #"output.conll2003_features"
    trainingfile =argv[4]
    inputfile = argv[5]
    outputfile = argv[6]
    language_model = argv[7] #"datas/GoogleNews-vectors-negative300.bin"

    selected_features = ["token","pos","tag","previous","latter","capitals","stemm","lemma"]
    training_features, gold_labels = extract_features_and_selected_labels(trainingfile_f, selected_features)
    test_features, test_gold_labels = extract_features_and_selected_labels(inputfile_f,selected_features)
    vec = DictVectorizer()
    
    model = MultinomialNB()
    features_vectorized = vec.fit_transform(training_features)
    model.fit(features_vectorized, gold_labels)
    features = vec.transform(test_features)
    prediction = model.predict(features)
    print(classification_report(test_gold_labels,prediction))
    print(confusion_matrix(test_gold_labels, prediction))
    
    
    model = LogisticRegression(max_iter=10000)
    model.fit(features_vectorized, gold_labels)
    features = vec.transform(test_features)
    prediction = model.predict(features)
    print(classification_report(test_gold_labels,prediction))
    print(confusion_matrix(test_gold_labels, prediction))

    training_features, gold_labels = extract_embeddings_as_features_and_gold(trainingfile, language_model)
    # X_test, y_test
    test_features, tests_gold_labels = extract_embeddings_as_features_and_gold(inputfile, language_model)
    model = LinearSVC(max_iter=10000)
    model.fit(training_features[:20000], gold_labels[:20000])
    prediction = model.predict(test_features)
    print(classification_report(tests_gold_labels,prediction))
    print(confusion_matrix(tests_gold_labels, prediction))
    param_grid = {'C':[1,10,100,1000], 'loss': ['hinge', 'squared_hinge'], 'penalty': ['l1', 'l2']}
    grid = GridSearchCV(LinearSVC(),param_grid,refit = True, verbose=2)
    grid.fit(training_features[:20000], gold_labels[:20000])
    grid.best_params_
    gold_labels
    predic = grid.predict(test_features)
    print(classification_report(tests_gold_labels,predic))
    print(confusion_matrix(tests_gold_labels, predic))

    main(system_type="with_features")
    main(system_type="word_embeddings")