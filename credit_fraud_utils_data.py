import pandas as pd
from numpy import where
import numpy as np
from utils.helper_fun import load_df, load_x_t, split_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler 
from imblearn.pipeline import Pipeline
from utils.helper_fun import load_config

config = load_config()
process_config = config['preprocessing']

RANDOM_STATE = config['random_state']


class Preprocessing:
    def __init__(self, df: pd.DataFrame):
        self.df=df

    def remove_duplicates(self):
        drop = process_config['remove_dublicates']
        if drop:
            self.df.drop_duplicates(inplace=True)
        return self.df

    def handle_outliers(self):
        drop = process_config['remove_outlier']
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
                self.df[col] = self.df[col].clip(lower, upper)

        return self.df

    def change_time_to_hours(self, col):
        self.df[col] = self.df[col] / 3600.0  
        return self.df


class Sampling():
    def __init__(self, x:pd.DataFrame,t:pd.Series):
        self.x=x
        self.t=t

    def under_sampling(self):
        factor = process_config['under_factor']
        counter = Counter(self.t)
        minority_count = counter[1]
        majority_count = counter[0]

        requested_majority = int(factor * minority_count)
        requested_majority = min(requested_majority, majority_count)

        rus = RandomUnderSampler(sampling_strategy={0: requested_majority,1: minority_count})
        ux, ut = rus.fit_resample(self.x, self.t)
        return ux, ut


    def over_sample(self):
        factor = process_config['over_factor']
        strategy = process_config['over_strategy']
        k_neig = process_config['k_neig']
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

    def under_over_sample(self):
        over_factor = process_config['over_factor']
        under_factor = process_config['under_factor']
        strategy = process_config['over_strategy']
        k_neig = process_config['k_neig']
        
        
        counter = Counter(self.t)
        majority_factor = counter[0]
        minority_factor = counter[1]

        if strategy == 'random':
            os = RandomOverSampler(random_state=RANDOM_STATE, sampling_strategy={1: majority_factor / over_factor})
        else:
            os = SMOTE(random_state=RANDOM_STATE,
                        sampling_strategy={1: int(minority_factor*over_factor)},
                        k_neighbors=k_neig)

        rus = RandomUnderSampler(sampling_strategy={0: int(majority_factor /under_factor)})

        pip = Pipeline(steps=[('over', os), ('under', rus)])
        uox, uot = pip.fit_resample(self.x, self.t)
        return uox, uot
    

class Processing_Pipeline():
    def __init__(self):
        pass

    def scaling(self, x):
        option = process_config['scaler_option']
        if option == 1:
            scaler = MinMaxScaler()
        elif option == 2:
            scaler = StandardScaler()
        elif option == 3:
            scaler = Normalizer()
        else:
            return None
        
        x = scaler.fit_transform(x)
        return scaler, x

    def apply_preprocessing(self, df:pd.DataFrame):
        remove_dublicate = process_config['remove_dublicates']
        remove_outlier = process_config['remove_outlier']
        change_time = process_config['change_time']

        preprocessing = Preprocessing(df)
        if remove_dublicate:
            df=preprocessing.remove_duplicates()
        if remove_outlier:
            df = preprocessing.handle_outliers()
        if change_time:
            df = preprocessing.change_time_to_hours('Time')

        return df

    def apply_sampling(self, x,t):
        sample_option = process_config['sample_option']
        sampling = Sampling(x,t)
        
        if sample_option==1:
            x,t = sampling.under_sampling()
        elif sample_option==2:
            x,t = sampling.over_sample()
        else:
            x,t = sampling.under_over_sample()
        return x,t

    def apply_scaling(self, x_train:pd.DataFrame, t_train:pd.Series, x_val=None, t_val=None, train_val=True):
        scaling_option = process_config['scaler_option']
        if train_val:
            scaler, x_trian_transformed = self.scaling(x_train)
            x_val_transformed = scaler.transform(x_val)
            return x_trian_transformed, t_train, x_val_transformed, t_val
        else:
            _,x_tranformed = self.scaling(x_train)
            return x_tranformed, t_train

