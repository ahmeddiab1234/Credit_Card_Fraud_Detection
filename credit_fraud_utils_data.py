import pandas as pd
from numpy import where
import numpy as np
from utils.helper_fun import load_df, load_x_t, split_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler 
from imblearn.pipeline import Pipeline

RANDOM_STATE=42


class Preprocessing:
    def __init__(self, x: pd.DataFrame):
        self.x=x

    def remove_duplicates(self, drop=True):
        if drop:
            self.x.drop_duplicates(inplace=True)
        return self.x

    def handle_outliers(self, drop=True):
        cols = self.x.columns
        for col in cols:
            q1 = self.x[col].quantile(0.25)
            q3 = self.x[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)

            if drop:
                self.x = self.x[(self.x[col] >= lower) & (self.x[col] <= upper)]
            else:
                self.x[col] = self.x[col].clip(lower, upper)

        return self.x

    def change_time_to_hours(self, col):
        self.x[col] = self.x[col] / 3600.0  
        return self.x


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

    def scaling(self, x, option=1):
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

    def apply_preprocessing(self, x:pd.DataFrame, remove_dublicate=True, remove_outlier=True, change_time=True):
        preprocessing = Preprocessing(x)
        if remove_dublicate:
            x=preprocessing.remove_duplicates()
        if remove_outlier:
            x = preprocessing.handle_outliers(True)
        if change_time:
            x = preprocessing.change_time_to_hours('Time')

        return x

    def apply_sampling(self, x,t, sample_option=1, under_factor=20, over_factor=20, over_strategy='smote'):
        sampling = Sampling(x,t)
        
        if sample_option==1:
            x,t = sampling.under_sampling(under_factor)
        elif sample_option==2:
            x,t = sampling.over_sample(over_factor)
        else:
            x,t = sampling.under_over_sample(under_factor, over_factor,over_strategy)
        return x,t

    def apply_scaling(self, x_train:pd.DataFrame, t_train:pd.Series, x_val=None, t_val=None, scaling_option=1, train_val=True):
        if train_val:
            scaler, x_trian_transformed = self.scaling(x_train, scaling_option)
            x_val_transformed = scaler.transform(x_val)
            return x_trian_transformed, t_train, x_val_transformed, t_val
        else:
            _,x_tranformed = self.scaling(x_train,scaling_option)
            return x_tranformed, t_train

