import pandas as pd
import pickle
data=pd.read_csv(r"C:\Users\shaik\Projects\IPL score prediction\model_building\finalipl.csv")
data.head()
data.dtypes
data["date"]=pd.to_datetime(data["date"])
X_train = data.drop(labels='total',axis=1)[data['date'].dt.year <= 2016]
X_test = data.drop(labels='total',axis=1)[data['date'].dt.year >=2017]
print(X_train.shape)
print(X_test.shape)
Y_train = data[data['date'].dt.year <= 2016]['total'].values
Y_test = data[data['date'].dt.year >= 2017]['total'].values
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)

# --- Model Building ---
# Gradient Boosting Model
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor(n_estimators=300,learning_rate=0.1,
 max_depth=4,
 min_samples_leaf=3,
 min_samples_split=2,
   random_state=10)
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
 
r2=r2_score(Y_test,Y_pred)
print("R-squared:",r2)
 
rmse=np.sqrt(mean_squared_error(Y_test,Y_pred))
print("RMSE:",rmse)
 
adjusted_r_squared =  1 - (1-r2)*((len(Y_train)+len(Y_test))-1)/((len(Y_train)+len(Y_test))-X_train.shape[1]-1)
print("Adj R-square:",adjusted_r_squared)
# Creating a pickle file for the regressor
filename = 'gb_model.pkl'
pickle.dump(regressor, open(filename, 'wb'))