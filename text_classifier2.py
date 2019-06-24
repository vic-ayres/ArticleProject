import numpy as np
import load_data as load
import spacy_tokenizer as st
import word_embeddings as emb
from gensim.models import KeyedVectors
from confusion_matrix import ConfusionMatrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


# Load in the data from annotated atricles
articles, labels = load.load_data()
print("Gathered vaild Data...")
print()


# Tokenizes each article and saves them in "docs"
# Remove puncutations, numbers, and whitespace
docs = []
for a in articles:
    doc = st.tokenize(a)
    docs.append(a)


# Each tokenized document is associated with the
# average of each word vector in the document
word_embedding_list = []
word_vectors = KeyedVectors.load('vectors.kv', mmap='r')
for d in docs:
    vector = emb.word_embedding(word_vectors, doc)
    word_embedding_list.append(vector)
    break
    
print(word_embedding_list)
print(len(word_embedding_list))
print()


# Splits the annotated data into training, developing, and testing data
# Stratifies them based on the labels
x, x_test, y, y_test = train_test_split(word_embedding_list, labels,
                                                    test_size=0.1,
                                                    stratify = labels)
x_train, x_dev, y_train, y_dev = train_test_split(x,y,
                                                  test_size=0.1,
                                                  stratify = y)
print("Data split to train, develop, and test...")
print()


# Oversamples the training data using SMOTE
'''smote = SMOTE(sampling_strategy = 'minority')
xtr, ytr = smote.fit_resample(x_train, y_train)
print('Oversamples data using SMOTE...')'''


# Create and train Mutlinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(x_train, y_train)
print("Classifier trained...")
print()


# Evaluate the predictive accuracy
predicted = clf.predict(x_dev)
print("Mean Accuracy: " + str(np.mean(predicted == y_dev)))
print()

# Prints out the confusion matrix
labels = ['fp_ic', 'fp_terr', 'fp_other', 'dom', 'notapp']
cm = ConfusionMatrix(labels, y_dev, predicted)
cm.print_matrix()
print("Actual: " + str(cm.count_actual()))
print("Predicted: "+ str(cm.count_predicted()))

# Prints the classification report
print(classification_report(y_dev, predicted, target_names=labels))
