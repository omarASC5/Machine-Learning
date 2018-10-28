import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
# Percent Volatility - Percent Change of stock throughout the day. How much the stock changed
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
# Daily Percent Change - Eng Stock - Open Stock
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Volume is amount of trades occured that day
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

print(df.head())