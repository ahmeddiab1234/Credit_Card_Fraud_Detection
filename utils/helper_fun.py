
from sklearn.model_selection import train_test_split
import pandas as pd
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
    x,y = x,t = df.iloc[:,:-1], df.iloc[:,-1]
    return df, x,y

def split_data(x,t, split_sz=0.2):
    x_train, x_val, t_train, t_val = train_test_split(x,t, test_size=split_sz, random_state=RANDOM_STATE)
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



if __name__=='__main__':
    df= load_df('data/split/train.csv')
    df,x,t=load_x_t(df)
    print(df.shape, x.shape, t.shape)
    x_train, x_val, t_train, t_val = split_data(x,t)
    print(df.shape, x.shape, t.shape)