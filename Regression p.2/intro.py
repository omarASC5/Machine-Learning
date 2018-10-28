import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
# Percent Volatility - Percent Change of stock throughout the day. How much the stock changed
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
# Daily Percent Change - Eng Stock - Open Stock
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Volume is amount of trades occured that day
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]


# Could be replaced with any column, not just stocks!
forecast_col = 'Adj. Close'
# We need to replace NaN data with something because we can't work with NaN data. This is better 
# than getting rid of data

df.fillna(-99999, inplace=True)

# Generally we use regression to forecast out
# Forecast 10% of the latest data
forecast_out = int(math.ceil(0.01*len(df)))

# Shifts a column negatively, upwards. Forecast_out parameter is the amount of columns to shift up by
#  Makes the forecast_col our label.
# Label is the Adj. Close predicted 10 days into the future
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
# Prints everything df
print(df.head())

X = np.array(df.drop(['label'], 1)) # Features are capital X, df.drop returns new data frame, then 
# it's converted to numpy array, then saved to X
# Labels are lowercase y
y = np.array(df['label'])

X = preprocessing.scale(X) # Scaling data before sending it into a classifier

y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train) # fit = train
accuracy = clf.score(X_test, y_test) # score = test
print(accuracy)

