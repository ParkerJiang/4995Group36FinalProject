#!/usr/bin/env python
# coding: utf-8

# This preprocessing procedure is by radekosmulski on Github/Kaggle.

# In[4]:


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/308635
def customer_hex_id_to_int(series):
    return series.str[-16:].apply(hex_id_to_int)

def hex_id_to_int(str):
    return int(str[-16:], 16)

def article_id_str_to_int(series):
    return series.astype('int32')

def article_id_int_to_str(series):
    return '0' + series.astype('str')

class Categorize(BaseEstimator, TransformerMixin):
    def __init__(self, min_examples=0):
        self.min_examples = min_examples
        self.categories = []
        
    def fit(self, X):
        for i in range(X.shape[1]):
            vc = X.iloc[:, i].value_counts()
            self.categories.append(vc[vc > self.min_examples].index.tolist())
        return self

    def transform(self, X):
        data = {X.columns[i]: pd.Categorical(X.iloc[:, i], categories=self.categories[i]).codes for i in range(X.shape[1])}
        return pd.DataFrame(data=data)


# In[9]:


def preprocAll():
    transactions = pd.read_csv('../../../../h-and-m-personalized-fashion-recommendations/transactions_train.csv', dtype={"article_id": "str"})
    customers = pd.read_csv('../../../../h-and-m-personalized-fashion-recommendations/customers.csv')
    articles = pd.read_csv('../../../../h-and-m-personalized-fashion-recommendations/articles.csv', dtype={"article_id": "str"})
    
    transactions['customer_id'] = customer_hex_id_to_int(transactions['customer_id'])
    transactions.t_dat = pd.to_datetime(transactions.t_dat, format='%Y-%m-%d')
    transactions['week'] = 104 - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7
    transactions.article_id = article_id_str_to_int(transactions.article_id)
    articles.article_id = article_id_str_to_int(articles.article_id)

    transactions.week = transactions.week.astype('int8')
    transactions.sales_channel_id = transactions.sales_channel_id.astype('int8')
    transactions.price = transactions.price.astype('float32')
    transactions.drop(columns='t_dat').info(memory_usage='deep')

    customers.customer_id = customer_hex_id_to_int(customers.customer_id)
    for col in ['FN', 'Active', 'age']:
        customers[col].fillna(-1, inplace=True)
        customers[col] = customers[col].astype('int8')

    customers.club_member_status = Categorize().fit_transform(customers[['club_member_status']]).club_member_status
    customers.postal_code = Categorize().fit_transform(customers[['postal_code']]).postal_code
    customers.fashion_news_frequency = Categorize().fit_transform(customers[['fashion_news_frequency']]).fashion_news_frequency

    for col in articles.columns:
        if articles[col].dtype == 'object':
            articles[col] = Categorize().fit_transform(articles[[col]])[col]

    for col in articles.columns:
        if articles[col].dtype == 'int64':
            articles[col] = articles[col].astype('int32')

    transactions.sort_values(['t_dat', 'customer_id'], inplace=True)

    transactions.to_parquet('transactions_train.parquet')
    customers.to_parquet('customers.parquet')
    articles.to_parquet('articles.parquet')

