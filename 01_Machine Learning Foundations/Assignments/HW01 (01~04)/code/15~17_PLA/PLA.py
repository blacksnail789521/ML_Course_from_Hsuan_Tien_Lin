import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
import os
from sklearn.utils import shuffle
from tqdm import trange



def load_training_data():
    
    X, y = [], []
    with open("PLA_train.dat", "r") as input_file:
        line_list = input_file.readlines()
        for line in line_list:
            data = line.replace("\n", "").split("\t")
            # For x, we need to insert x0 as 1 because w0 will be b.
            x = [ float(element) for element in data[0].split(" ") ]
            x.insert(0, 1)
            X.append(x)
            y.append(int(data[1]))
        
    return np.array(X), np.array(y)


class PLA():
    
    def __init__(self, learning_rate = 1):
        
        self.learning_rate = learning_rate
        self.update_count = 0
        
    
    def set_parameters(self, X):
        
        self.m = X.shape[0]
        self.n = X.shape[1]
        
        self.W = np.zeros((self.n, 1))
        
    
    def get_y_head(self, x, W):
        
        return 1 if np.dot(W.T, x)[0][0] > 0 else -1
        
    
    def fit(self, X, y):
        
        self.set_parameters(X)
        
        correct_count = 0
        index = 0
        while 1:
            x = X[index].T.reshape(self.n, 1)
            if self.get_y_head(x, self.W) != y[index]:
                # This training example is incorrect with current W.
                self.W = self.W + self.learning_rate * y[index] * x
                correct_count = 0
                self.update_count += 1
            else:
                # This training example is correct with current W.
                correct_count += 1
                if correct_count == self.m:
                    break
            
            # Update index.
            index = 0 if index == self.m - 1 else index + 1



X, y = load_training_data()

""" Built-in PLA in sklearn. """
"""--------------------------------------"""
# clf = Perceptron(tol=1e-3, random_state=0)
# clf.fit(X, y)
# score = clf.score(X, y)
"""--------------------------------------"""

question_number = 15
# question_number = 16
# question_number = 17

if question_number == 15:
    n_times = 1 # We don't shuffle the data.
    learning_rate = 1

elif question_number == 16:
    n_times = 2000
    learning_rate = 1

elif question_number == 17:
    n_times = 2000
    learning_rate = 0.5

    
update_count_list = []
for i in trange(n_times):
    if n_times != 1:
        X, y = shuffle(X, y, random_state = i)
    clf = PLA(learning_rate)
    clf.fit(X, y)
    update_count_list.append(clf.update_count)
print("")
print("average of update_count:", np.average(update_count_list))