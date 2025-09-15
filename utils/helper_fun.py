
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
import pickle

import matplotlib
matplotlib.use("Qt5Agg") 

import matplotlib.pyplot as plt

RANDOM_STATE=42

def load_df(path):
    if path is None:
        raise ValueError('Path is don`t defined correctly')
    df = pd.read_csv(path)
    
    return df
    
def load_x_t(df:pd.DataFrame):
    x,y = df.iloc[:,:-1], df.iloc[:,-1]
    return df, x,y

def split_data(x,t, split_sz=0.2):
    x_train, x_val, t_train, t_val = train_test_split(x,t, test_size=split_sz, random_state=RANDOM_STATE, stratify=t)
    return x_train, x_val, t_train, t_val

def visualize_pr(precision, recall, threshall, versus_threshall=True):
        if versus_threshall:
            plt.plot(threshall, precision, 'b--', label='Precession')
            plt.plot(threshall, recall, 'g--', label='Recall')
            plt.xlabel('Threshall')
            plt.legend(loc='upper right')
        else:
            plt.plot(precision, recall, 'g--')
            plt.xlabel('Precision')
            plt.ylabel('Recall')
        plt.show()

def save_model(model, threshold, model_save_name):
    model_dict = {
        "model": model,
        "threshold": threshold,
        "model_name": model_save_name
    }
    root_dir = os.getcwd()

    with open(os.path.join(root_dir, 'model.pkl'), 'wb') as file:
        pickle.dump(model_dict, file)

def load_model():
    with open("model.pkl", 'rb') as f:
        loaded_dict = pickle.load(f)

    return loaded_dict['model'], loaded_dict['threshold'], loaded_dict['model_name']


def save_model_knn(model, threshold, model_save_name):
    model_dict = {
        "model_knn": model,
        "threshold": threshold,
        "model_name": model_save_name
    }
    root_dir = os.getcwd()

    with open(os.path.join(root_dir, 'model_knn.pkl'), 'wb') as file:
        pickle.dump(model_dict, file)


def load_model_knn():
    with open("model_knn.pkl", 'rb') as f:
        loaded_dict = pickle.load(f)

    return loaded_dict['model_knn'], loaded_dict['threshold'], loaded_dict['model_name']


if __name__=='__main__':
    df= load_df('data/split/train.csv')
    df,x,t=load_x_t(df)
    print(df.shape, x.shape, t.shape)
    x_train, x_val, t_train, t_val = split_data(x,t)
    print(df.shape, x.shape, t.shape)

    