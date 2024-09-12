import streamlit as st








with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

# 2. Load the OneHot Encoder
with open('onehot_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# 3. Load the Regressor Model
with open('regressor_model.pkl', 'rb') as file:
    GB = pickle.load(file)
# Initialize the LabelEncoder


# Columns to be label encoded
columns_to_label_encode = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type']

# Apply label encoding
for col in columns_to_label_encode:
    test_data[col] = le.fit_transform(test_data[col])

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, dtype=int)  # `dtype=int` ensures the output is in integers

# Columns to be one-hot encoded
columns_to_one_hot_encode = ['Item_Type', 'Outlet_Type', 'Outlet_Identifier']

# Apply the encoder
encoded_data = encoder.fit_transform(test_data[columns_to_one_hot_encode])

# Create a DataFrame with the encoded data
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns_to_one_hot_encode))

# Drop the original columns that were one-hot encoded
test_data.drop(columns=columns_to_one_hot_encode, inplace=True)

# Concatenate the encoded columns back to the original DataFrame
test_data = pd.concat([test_data, encoded_df], axis=1)

# Check the resulting DataFrame
test_data.drop(columns=['Item_Identifier'], inplace=True)
test_data.head()

