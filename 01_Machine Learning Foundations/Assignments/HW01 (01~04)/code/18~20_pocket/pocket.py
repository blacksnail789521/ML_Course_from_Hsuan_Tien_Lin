import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
import os
from sklearn.utils import shuffle
from tqdm import trange



def load_training_data():
    
    X, y = [], []
    with open("pocket_train.dat", "r") as input_file:
        line_list = input_file.readlines()
        for line in line_list:
            data = line.replace("\n", "").split("\t")
            # For x, we need to insert x0 as 1 because w0 will be b.
            x = [ float(element) for element in data[0].split(" ") ]
            x.insert(0, 1)
            X.append(x)
            y.append(int(data[1]))
        
    return np.array(X), np.array(y)


def load_testing_data():
    
    X, y = [], []
    with open("pocket_test.dat", "r") as input_file:
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
    
    def __init__(self, learning_rate = 1, update_count_upper_limit = 50):
        
        self.learning_rate = learning_rate
        self.update_count_upper_limit = update_count_upper_limit
        self.update_count = 0
        
    
    def set_parameters(self, X):
        
        self.m = X.shape[0]
        self.n = X.shape[1]
        
        self.W = np.zeros((self.n, 1))
        self.pocket_W = self.W.copy()
        
    
    def get_y_head(self, x, W):
        
        return 1 if np.dot(W.T, x)[0][0] > 0 else -1
    
    
    def calculate_overall_correct_count(self, X, y, W):
        
        correct_count = 0
        for index in range(self.m):
            x = X[index].T.reshape(self.n, 1)
            if self.get_y_head(x, W) == y[index]:
                correct_count += 1
        
        return correct_count
        
    
    def fit(self, X, y):
        
        self.set_parameters(X)
        
        index = np.random.randint(self.m)
        while self.update_count < self.update_count_upper_limit:
            x = X[index].T.reshape(self.n, 1)
            if self.get_y_head(x, self.W) != y[index]:
                # This training example is incorrect with current W.
                self.W = self.W + self.learning_rate * y[index] * x
                self.update_count += 1
                
                # Check that if we need to update pocket_W with current W.
                if self.calculate_overall_correct_count(X, y, self.W) > \
                   self.calculate_overall_correct_count(X, y, self.pocket_W):
                    self.pocket_W = self.W.copy()
            
            # Update index.
            index = np.random.randint(self.m)
    
    
    def score(self, X, y, current_W_or_pocket_W = "pocket_W"):
        
        if current_W_or_pocket_W == "pocket_W":
            target_W = self.pocket_W
        elif current_W_or_pocket_W == "current_W":
            target_W = self.W
        
        return (self.m - self.calculate_overall_correct_count(X, y, target_W)) / self.m



X_train, y_train = load_training_data()
X_test, y_test = load_testing_data()


question_number = 18
# question_number = 19
# question_number = 20

if question_number == 18:
    update_count_upper_limit = 50
    current_W_or_pocket_W = "pocket_W"

elif question_number == 19:
    update_count_upper_limit = 50
    current_W_or_pocket_W = "current_W"

elif question_number == 20:
    update_count_upper_limit = 100
    current_W_or_pocket_W = "pocket_W"

    
testing_score_list = []
for i in trange(2000):
    np.random.seed(i)
    clf = PLA(update_count_upper_limit = update_count_upper_limit)
    clf.fit(X_train, y_train)
    testing_score_list.append(clf.score(X_test, y_test, current_W_or_pocket_W))
print("")
print("average of testing_score:", np.average(testing_score_list))