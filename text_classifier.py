import numpy as np
import load_data as load
import spacy_tokenizer as st
from confusion_matrix import ConfusionMatrix
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC



# Load in the data from annotated atricles
articles, labels = load.load_data()
print("Gathered vaild Data...")
print()



# Make the articels into a feature vector using a bag of words
# and oversamples using SMOTE
count_vect = CountVectorizer(tokenizer = st.tokenize, ngram_range=(1,2), min_df = 0)
articles_vect = count_vect.fit_transform(articles)
print("Shape of the data: " + str(articles_vect.shape))





# Splits the annotated data into training and testing data
# Stratifies them based on the labels
X_train, X_test, y_train, y_test = train_test_split(articles_vect, labels,
                                                    test_size=0.2,
                                                    stratify = labels)
print("Data split to training and test...")
print()

print(np.array(X_test).size)
# Cross validation with a StratifiedKFold
# Has 5 folds
print("Cross validation with stratified folds:")
kf = StratifiedKFold(n_splits=5,shuffle=True)
clf_score =[]
lr_score = []
rfc_score = []
svc_score = []
i=1
for train_index,test_index in kf.split(X_train,y_train):

    xtr, ytr = X_train[train_index], y_train[train_index]
    smote = SMOTE(sampling_strategy = 'minority')
    xtr, ytr = smote.fit_resample(xtr, ytr)
    print(xtr.shape)
    xdev, ydev = X_train[test_index], y_train[test_index]
    
    #model 1
    clf = MultinomialNB()
    clf.fit(xtr,ytr)
    clf_score.append(accuracy_score(ydev,clf.predict(xdev)))

    #model 2
    lr = LogisticRegression(solver='lbfgs',
                            multi_class='multinomial')
    lr.fit(xtr,ytr)
    lr_score.append(accuracy_score(ydev,lr.predict(xdev)))

    #model 3
    rfc = RandomForestClassifier(n_estimators = 100)
    rfc.fit(xtr, ytr)
    rfc_score.append(accuracy_score(ydev, rfc.predict(xdev)))

    #model 4
    svc = SVC(gamma = 'scale')
    svc.fit(xtr,ytr)
    svc_score.append(accuracy_score(ydev,svc.predict(xdev)))
    
    i+=1
print()

# Mean score of the cross validation for different Models
print("Mean score of Naive Bayes cross validations: " + str(np.mean(clf_score)))
print("Mean score of Logistic Regression cross validations: " + str(np.mean(lr_score)))
print("Mean score of Random Forest Classifier cross validations: " + str(np.mean(rfc_score)))
print("Mean score of SVC cross validations: " + str(np.mean(svc_score)))

print('Classifier created...')
print()



xtr, xdev, ytr, ydev = train_test_split(X_train, y_train,
                            test_size=0.2,
                            stratify = y_train)
smote = SMOTE(sampling_strategy = 'minority')
xtr, ytr = smote.fit_resample(xtr, ytr)

clf.fit(xtr, ytr)
print("Classifier trained...")
print()

predicted = clf.predict(xdev)
# Evaluate the predictive accuracy
print("Mean Accuracy: " + str(np.mean(predicted == ydev)))
print()

# Prints out the confusion matrix
labels = ['fp_ic', 'fp_terr', 'fp_other', 'dom', 'notapp']
cm = ConfusionMatrix(labels, ydev, predicted)
cm.print_matrix()
print("Actual: " + str(cm.count_actual()))
print("Predicted: "+ str(cm.count_predicted()))

# Prints the classification report
print(classification_report(ydev, predicted, target_names=labels))
