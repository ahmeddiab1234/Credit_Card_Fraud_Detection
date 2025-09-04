"""

loading data
processing:
    - remove dublicates 
    - remove outliers
    - transform time to hours -> /60/60
    - preprocessing (standarscaling, minmax sacling)

    - under sampling (zeros)
    - over sampling (ones) -> random, smote
    - over sampling + under sampling

    - Cost-Sensitive Learning, change weights for two classes
"""
import pandas as pd
from numpy import where
from utils.helper_fun import load_df, load_x_t, split_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler 

RANDOM_STATE=42

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline

RANDOM_STATE = 42

class Preprocessing:
    def __init__(self, df: pd.DataFrame):
        self.df=df

    def remove_duplicates(self, drop=True):
        if drop:
            self.df.drop_duplicates(inplace=True)
        return self.df

    def handle_outliers(self, drop=True):
        cols = self.df.columns[:-1]
        for col in cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)

            if drop:
                self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
            else:
                self.df[col] = np.where(
                    self.df[col] < lower, lower,
                    np.where(self.df[col] > upper, upper, self.df[col])
                )
        return self.df

    def change_time_to_hours(self, col):
        self.df[col] = self.df[col] / 3600.0  
        return self.df

    def scaling(self, option=1):
        if option == 1:
            scaler = MinMaxScaler()
        elif option == 2:
            scaler = StandardScaler()
        elif option == 3:
            scaler = Normalizer()
        else:
            return None
        cols = self.df.columns[:-1]
        
        df[cols] = scaler.fit_transform(df[cols])
        return scaler, self.df


class Sampling():
    def __init__(self, x:pd.DataFrame,t:pd.Series):
        self.x=x
        self.t=t

    def under_sampling(self, factor=50):
        counter = Counter(self.t)
        minority_count = counter[1]
        majority_count = counter[0]

        requested_majority = int(factor * minority_count)
        requested_majority = min(requested_majority, majority_count)

        rus = RandomUnderSampler(sampling_strategy={0: requested_majority,1: minority_count})
        ux, ut = rus.fit_resample(self.x, self.t)
        return ux, ut


    def over_sample(self, factor=50, strategy='smote', k_neig=3):
        counter = Counter(self.t)
        majority_factor = counter[0]
        if strategy == 'random':
            os = RandomOverSampler(random_state=RANDOM_STATE, sampling_strategy={1: int(majority_factor / factor)})
        else:
            os = SMOTE(random_state=RANDOM_STATE,
                        sampling_strategy={1: int(majority_factor / factor)},
                        k_neighbors=k_neig)
        ox, ot = os.fit_resample(self.x, self.t)
        return ox, ot

    def under_over_sample(self, under_factor=2, over_factor=20, strategy='smote', k_neig=3):
        counter = Counter(self.t)
        majority_factor = counter[0]

        if strategy == 'random':
            os = RandomOverSampler(random_state=RANDOM_STATE, sampling_strategy={1: majority_factor / over_factor})
        else:
            os = SMOTE(random_state=RANDOM_STATE,
                        sampling_strategy={1: int(majority_factor / over_factor)},
                        k_neighbors=k_neig)

        rus = RandomUnderSampler(sampling_strategy={0: int(majority_factor / under_factor)})

        pip = Pipeline(steps=[('over', os), ('under', rus)])
        uox, uot = pip.fit_resample(self.x, self.t)
        return uox, uot



    

class Processing_Pipeline():
    def __init__(self):
        pass

    def apply_preprocessing(self,path, scaling_option=1, scaler=True, val=False, remove_dublicate=True, remove_outlier=True, change_time=True):
        df = load_df(path)
        preprocessing = Preprocessing(df)
        if remove_dublicate:
            df=preprocessing.remove_duplicates()
        if remove_outlier:
            df = preprocessing.handle_outliers(True)
        if change_time:
            df = preprocessing.change_time_to_hours('Time')

        return df

    def apply_sampling(self,sample_option=1, under_factor=20, over_factor=20, over_strategy='smote'):
        df = self.apply_preprocessing()
        x,t = load_x_t(df)
        sampling = Sampling()
        
        if sample==1:
            x,t = sampling.under_sampling(under_factor)
        elif sample==2:
            x,t = sampling.over_sample(over_factor)
        else:
            x,t = sampling.under_over_sample(under_factor, over_factor,over_strategy)





