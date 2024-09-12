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
from src.utils import evaluate_model

## Data Manipulation
data = pd.read_csv('./../artifacts/Train.csv')
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
data.drop(columns=['Item_Identifier'], inplace=True)

from sklearn.model_selection import train_test_split
X = data.drop('Item_Outlet_Sales',axis=1)
y = data['Item_Outlet_Sales']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(), 
    "AdaBoost Regressor": AdaBoostRegressor()
}
model_list = []
r2_list =[]

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate Train and Test dataset
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    
    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    r2_list.append(model_test_r2)
    
    print('='*35)
    print('\n')

GB= GradientBoostingRegressor()
GB.fit(X_train, y_train)
test_data = pd.read_csv('../artifacts/Test.csv')
test_data.head()

# Save Label Encoder to .pkl
with open('../artifacts/label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)

# Save OneHot Encoder to .pkl
with open('../artifacts/onehot_encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)

# Save Regressor Model to .pkl
with open('../artifacts/regressor_model.pkl', 'wb') as file:
    pickle.dump(GB, file)
