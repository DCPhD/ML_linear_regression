import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

# change diplay output settings for database
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 200)

# read in file
dataset = pd.read_csv('combined_dataset.csv')

# check for NaN values
#print(dataset.isnull().sum())

# scatterplot to evaluate relationship between variables
dataset.plot(x='PROSPECTS', y='SALES', style='o')
plt.title('PROSPECTS vs SALES')
plt.xlabel('PROSPECTS')
plt.ylabel('SALES')
plt.show()

# attribute
X = dataset['PROSPECTS'].values.reshape(-1,1)

# make y target variable
y = dataset['SALES'].values.reshape(-1,1)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# import linear regression class
regressor = linear_model.LinearRegression()

# train the algorithm
regressor.fit(X_train, y_train)

# make prediction
y_pred = regressor.predict(X_test)

# compare prediction with actual
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
#print(df)

# check accuracy
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# OUTPUT FROM VARIOUS ATTRIBUTES
'''                     MAE
DAY                     285
VISITS                  292
MEMBERS                 285
TOTAL_CONVERSION        234
TOTAL PROSPECTS         225 * lowest mean error= most accurate predictor of sales
NEW PROSPECTS           342

'''