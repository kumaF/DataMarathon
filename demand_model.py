import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from sklearn.model_selection import train_test_split

# import data files
retailer_data = pd.DataFrame(pd.read_csv('Retailer.csv'))
product_data = pd.DataFrame(pd.read_csv('Product.csv'))
sales_data1 = pd.DataFrame(pd.read_csv('File 01 - Sales Data.csv'))
sales_data2 = pd.DataFrame(pd.read_csv('File 02 - Sales Data.csv'))

df = [ sales_data1, sales_data2]
data = pd.DataFrame(pd.concat(df)).sort_values(by=['Date'], ascending=False).dropna().drop_duplicates()

################### perform Train Test split on Data #####################
y_data = data['Net Sales Qty']
x_data = data.drop('Net Sales Qty', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3 ,random_state=101)

#print(len(x_train), len(x_test))
from sklearn.preprocessing import MinMaxScaler
#categorical cols
# product_code = tf.feature_column.categorical_column_with_hash_bucket(data['Product Code'], hash_bucket_size=1000)
# sales_rep = tf.feature_column.categorical_column_with_hash_bucket(data['Rep'], hash_bucket_size=1000)
# retailer_code = tf.feature_column.categorical_column_with_hash_bucket(data['Retailer Code'], hash_bucket_size=8000)


# feat_cols = [ product_code, sales_rep, retailer_code]

# input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train, batch_size=100, num_epochs=1000, shuffle=True,queue_capacity=1000,num_threads=1,target_column='target')

# model = tf.estimator.LinearClassifier(feature_columns=feat_cols)

# model.train(input_fn=input_func, steps=10000)

print('hello')