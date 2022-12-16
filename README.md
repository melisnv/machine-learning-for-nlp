# Machine Learning for NLP 

The following repository is consisted of seven main python files:
- basic_system.ipynb
- basic_evaluation.ipynb
- data_exploration_with_NER.ipynb
- feature_extraction.py
- features_ablation_analysis.ipynb
- ner_machine_learning.py
- lstm-ner.ipynb
- bert_finetuner.ipynb

In these files in order to run this, you will need to have the python installed.
In addition, you will need to install specific packages mentioned in the notebooks.
There are not extra libraries that you need to install.

### Important note:

"basic_evaluation.ipynb" -> this file is provides an overview of basic evaluation metrics like precision, recall, f-score and a confusion matrix from documents provided in the conll format. 

"basic_system.ipynb" -> this notebook provides code for implementing a very simple machine learning system for named entity recognition. Also this notebook includes the base experiment with Logistic Regression with the default features.

"data_exploration_with_NER.ipynb" -> this notebook helps to understand the distribution of the data in the training and development set.

"feature_extraction.py" -> this notebook extracts features like previous and latter tokens,stemming of the tokens, lemmatization of the tokens, and lastly whether tokens are capitalized or not. Later this new featured data will feed in three different models (NB,SVM and LR) for a performance comparison.

"features_ablation_analysis.ipynb" -> this notebook examines different features and combination of features' effect on the model performance.

"ner_machine_learning.py" -> this notebook performs SVM with embeddings and trains three different models (NB,SVM and LR) with the extracted features from the file feature_extraction.py.

"lstm-ner.ipynb" -> this notebook runs LSTM model on the conll data with default features, to make a comparison with base model's performance.

"bert_finetuner.ipynb" -> this notebook performs BERT model on the conll data to make a comparison with base model's performance.

### How to run

To run ner_machine_learning.py:

> python ner_machine_learning.py ‘path to train extracted features files’ ‘ path to dev extracted features file’ ‘path to output file for predictions of models trained on extracted features’ ‘ path to train file initial’ ‘ path to dev file initial’ ‘ path to output file for predictions of svm using word embeddings’  ‘ path to language model’

To run analyze_distribution.py

> python analyze_distribution.py ‘path to training file’ ‘path to dev file’ ‘ path to output file’ 

To run feature_extraction.py

> python feature-extraction.py ‘ path to training file’ ‘ path to dev file’ ‘ path to output file’ 

