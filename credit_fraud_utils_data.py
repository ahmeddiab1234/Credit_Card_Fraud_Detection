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
from utils.helper_fun import load_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler 

RANDOM_STATE=42

class Preprocessing():
    def __init__(self, x:pd.DataFrame,y:pd.DataFrame):
        self.x=x
        self.y=y

    def remove_dublicates(self, drop=True):
        if drop:
            self.x.drop_duplicates(inplace=True)
        return x
    
    def handle_outliers(self, drop=True):
        for col in self.x.columns:
            q1 = self.x[col].quantile(0.25)
            q3 = self.x[col].quantile(0.75)
            iqr = q3-q1

            lower = q1-(1.5*iqr)
            upper = q3+(1.5*iqr)

            if drop:
                self.x = self.x[(df[col]>=lower)|(self.x[col]<=upper)]
            else:
                self.x = where(self.x[col]<lower, lower, where(self.x[col]>upper, upper, self.x[col]))
        return x


    def change_time_to_hours(self, col):
        self.x[col] = self.x[col]/60/60
        return x


    def scaling(self, option=1):
        if option==1:
            scaler = MinMaxScaler()
        elif option==2:
            scaler = StandardScaler()
        elif option==3:
            scaler = Normalizer()
        else:
            return None
        
        x_transformed = scaler.fit_transform(self.x)
        return scaler, x_transformed
    

    def under_sampling(self, factor=50):
        counter = Counter(self.y)
        minority_factor = counter[1]
        rus = RandomUnderSampler(sampling_strategy={0:factor*minority_factor})
        ux, uy= rus.fit_resample(self.x, self.y)
        return ux, uy
    

    def over_sample(self, factor=50, strategy='smote', k_neig=3):
        counter = Counter(self.y)
        majority_factor = counter[0]
        if strategy=='random':
            os = RandomOverSampler(random_state=RANDOM_STATE,sampling_strategy={1:majority_factor/factor})
        else:
            os = SMOTE(random_state=RANDOM_STATE, sampling_strategy={1:majority_factor/factor}, k_neighbors=k_neig)

        ox,oy = os.fit_resample(self.x, self.y)
        return ox, oy
    

    def under_over_sample(self, under_factor=2, over_factor=20, strategy='smote', k_neig=3):
        counter = Counter(self.y)
        majority_factor = counter[0]

        if strategy=='random':
            os = RandomOverSampler(random_state=RANDOM_STATE,sampling_strategy={1:majority_factor/over_factor})
        else:
            os = SMOTE(random_state=RANDOM_STATE, sampling_strategy={1:majority_factor/over_factor}, k_neighbors=k_neig)

        rus = RandomUnderSampler(sampling_strategy={0:majority_factor/under_factor})

        from imblearn.pipeline import Pipeline
        pip = Pipeline(steps=[('over',os),('under', rus)])

        uox, uoy = pip.fit_resample(self.x, self.y)
        return uox, uoy




if __name__=='__main__':
    df,x,t = load_data('data/split/train.csv',True)
    print(Counter(t))  #Counter({0: 170579, 1: 305})
    print(x.shape, t.shape)
    preprocessing = Preprocessing(x,t)
    x = preprocessing.remove_dublicates(True)
    print(x.shape, t.shape)
    x = preprocessing.change_time_to_hours('Time')
    print(x.shape, t.shape)
    x = preprocessing.handle_outliers(False)
    print(x.shape, t.shape)

