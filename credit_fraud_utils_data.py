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
from utils.helper_fun import load_data
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler 

