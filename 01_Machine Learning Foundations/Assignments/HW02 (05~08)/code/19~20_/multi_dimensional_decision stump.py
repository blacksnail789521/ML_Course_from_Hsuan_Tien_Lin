import numpy as np
from tqdm import trange


def get_data():
    
    x_train, y_train = [], []
    with open("hw2_train.dat", "r") as input_file:
        line_list = input_file.readlines()
        for line in line_list:
            data = line.strip().split(" ")
            x = [ float(data[i]) for i in range(len(data) - 1) ]
            x_train.append(x)
            y_train.append(int(data[-1]))
        
    x_test, y_test = [], []
    with open("hw2_test.dat", "r") as input_file:
        line_list = input_file.readlines()
        for line in line_list:
            data = line.strip().split(" ")
            x = [ float(data[i]) for i in range(len(data) - 1) ]
            x_test.append(x)
            y_test.append(int(data[-1]))
    
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def train_model(x_train, y_train):
    
    s, theta, dimension, E_in, E_out = None, None, None, np.inf, None
    for dimension_candidate in range(x_train.shape[1]):
        x_train_with_single_dimension = x_train[ : , dimension_candidate]
        for s_candidate in [1, -1]:
            for theta_candidate in [ (x_train_with_single_dimension[i] + x_train_with_single_dimension[i+1]) / 2 for i in range(len(x_train_with_single_dimension) - 1) ]:
                hypothetis = lambda x : s_candidate * np.sign(x - theta_candidate)
                E_in_candidate = np.average([ 1 if hypothetis(x_train_with_single_dimension[i]) != y_train[i] else 0 \
                                              for i in range(len(x_train_with_single_dimension)) ])
                if E_in_candidate < E_in:
                    s = s_candidate
                    theta = theta_candidate
                    dimension = dimension_candidate
                    E_in = E_in_candidate
                    E_out = 0.5 + 0.3 * s * (np.abs(theta) - 1)
    
    return s, theta, dimension, E_in, E_out


def main():
    
    x_train, y_train, x_test, y_test = get_data()
        
    s, theta, dimension, E_in, E_out = train_model(x_train, y_train)
    print('s:', s)
    print('theta:', theta)
    print('dimension:', dimension)
    print('-----------------')
    print('E_in:', E_in)
    
    hypothetis = lambda x : s * np.sign(x - theta)
    E_out_for_testing_data = \
        np.average([ 1 if hypothetis(x_test[i, dimension]) != y_test[i] else 0 \
                     for i in range(len(x_test[:, dimension])) ])
    print('E_out:', E_out_for_testing_data)
    


if __name__ == '__main__':
    
    result = main()