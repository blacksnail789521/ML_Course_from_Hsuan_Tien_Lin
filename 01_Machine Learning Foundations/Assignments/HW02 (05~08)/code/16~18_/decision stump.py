import numpy as np
from tqdm import trange


def get_data():

    x_train = np.sort(np.random.uniform(-1, 1, 20))
    y_train = np.sign(x_train)
    for i, y in enumerate(y_train):
        if np.random.random() > 0.8:
            # Flip with p = 0.2
            y_train[i] *= -1
    
    return x_train, y_train


def train_model(x_train, y_train):
    
    s, theta, E_in, E_out = None, None, np.inf, None
    for s_candidate in [1, -1]:
        for theta_candidate in [ (x_train[i] + x_train[i+1]) / 2 for i in range(len(x_train) - 1) ]:
            hypothetis = lambda x : s_candidate * np.sign(x - theta_candidate)
            E_in_candidate = np.average([ 1 if hypothetis(x_train[i]) != y_train[i] else 0 \
                                          for i in range(len(x_train)) ])
            if E_in_candidate < E_in:
                s = s_candidate
                theta = theta_candidate
                E_in = E_in_candidate
                E_out = 0.5 + 0.3 * s * (np.abs(theta) - 1)
    
    return s, theta, E_in, E_out


def main():
    
    n_iterations = 5000
    
    E_in_list, E_out_list = [], []
    for i in trange(n_iterations):
        x_train, y_train = get_data()
        s, theta, E_in, E_out = \
            train_model(x_train, y_train)
        
        E_in_list.append(E_in)
        E_out_list.append(E_out)
    
    print()
    print('average of E_in:', np.average(E_in_list))
    print('average of E_out:', np.average(E_out_list))


if __name__ == '__main__':
    
    result = main()