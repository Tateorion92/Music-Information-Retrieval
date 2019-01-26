
# coding: utf-8

# In[1]:

import itertools
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB

def print_matrix(matrix, labels):
    result = [[0 for x in range(len(matrix)+2)] for y in range(len(matrix)+1)]
    for i in range(len(matrix)):
        matrix[i].append('|')
        attribute = chr(ord('a')+i)
        matrix[i].append(attribute+' = '+str(int(labels[i])))
        result[0][i] = attribute
        result[i+1] = matrix[i]
    result[0][-2] = ' '
    result[0][-1] = '<-- classified as'

    s = [[str(e) for e in row] for row in result]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '  '.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print "\n === Confusion Matrix ==="
    print '\n'.join(table)


# In[2]:

X, Y = load_svmlight_file("genres3.libsvm")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

classifiers = [DecisionTreeClassifier(max_depth=5),
                KNeighborsClassifier(3),
                BernoulliNB() ]

for i in range(len(classifiers)):
    classifier = classifiers[i]
    accuracy = classifier.fit(X_train, Y_train)
    Y_pred = accuracy.predict(X_test)
    accuracy = accuracy.score(X_test, Y_test)*100
    m = confusion_matrix(Y_test, Y_pred)

    print_matrix(m.tolist(), classifier.classes_)
    print('\n Correctly Classified Instances: \t {0} \t {1}%').format(int(round((float(accuracy)/100)*X_test.shape[0])), "%.2f" % accuracy)


# In[ ]:



