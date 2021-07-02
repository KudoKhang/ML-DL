from __future__ import print_function
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# train data 
d1 = [2,1,1,0,0,0,0,0,0]
d2 = [1,1,0,1,1,0,0,0,0]
d3 = [0,1,0,0,1,1,0,0,0]
d4 = [0,1,0,0,0,0,1,1,1]

train_data = np.array([d1, d2, d3 , d4])
label = np.array(['B', 'B', 'B', 'N'])

# test data
d5 = np.array([[2,0,0,1,0,0,0,1,0]])
d6 = np.array([[0,1,0,0,0,0,0,1,1]])
d7 = np.array([[1,3,0,1,0,0,0,0,0]])

model = MultinomialNB()

model.fit(train_data, label)

print(str(model.predict_proba(d6)[0]))