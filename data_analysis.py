import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# import data files
retailer_data = pd.DataFrame(pd.read_csv('Retailer.csv'))
product_data = pd.DataFrame(pd.read_csv('Product.csv'))
sales_data1 = pd.DataFrame(pd.read_csv('File 01 - Sales Data.csv'))
sales_data2 = pd.DataFrame(pd.read_csv('File 02 - Sales Data.csv'))

df = [ sales_data1, sales_data2]
data = pd.DataFrame(pd.concat(df)).sort_values(by=['Date'], ascending=False)

prod = data.groupby(data['Product Code']).count().soe

plt.bar()
year_wise = data.groupby()

ts  = pd.Series

print(prod.head())

retailer_data.dropna