import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


## Data Manipulation
data = pd.read_csv('../../artifacts/Train.csv')
data['Item_Weight'].fillna(data['Item_Weight'].median(), inplace=True)
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)


# Initialize the LabelEncoder
le = LabelEncoder()

# Columns to be label encoded
columns_to_label_encode = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type']

# Apply label encoding
for col in columns_to_label_encode:
    data[col] = le.fit_transform(data[col])



encoder = OneHotEncoder(sparse_output=False, dtype=int)  # `dtype=int` ensures the output is in integers
columns_to_one_hot_encode = ['Item_Type', 'Outlet_Type', 'Outlet_Identifier']

# Apply the encoder
encoded_data = encoder.fit_transform(data[columns_to_one_hot_encode])

# Create a DataFrame with the encoded data
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns_to_one_hot_encode))

# Drop the original columns that were one-hot encoded
data.drop(columns=columns_to_one_hot_encode, inplace=True)

# Concatenate the encoded columns back to the original DataFrame
data = pd.concat([data, encoded_df], axis=1)

# Check the resulting DataFrame
print(data.head())
