
from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_STATE=42

def load_data(path, train_only=True, split_sz=0.2):
    if path is None:
        raise ValueError('Path is don`t defined correctly')
    df = pd.read_csv(path)
    x,t = df.iloc[:,:-1], df.iloc[:,-1]
    
    if train_only:
        return df, x,t

    else:
        x_train, x_val, t_train, t_val = train_test_split(x,t, test_size=split_sz, random_state=RANDOM_STATE)
        return df, x_train, x_val, t_train, t_val
    


if __name__=='__main__':
    df,x,t = load_data('data/split/train.csv',True)
    print(df.shape, x.shape, t.shape)