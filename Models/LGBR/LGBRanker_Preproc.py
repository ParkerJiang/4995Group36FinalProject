#!/usr/bin/env python
# coding: utf-8

# In[1]:

# This file is by ALEX VISHNEVSKIY from
# https://www.kaggle.com/code/alexvishnevskiy/ranking-item-features/notebook
# https://www.kaggle.com/code/alexvishnevskiy/ranking-user-features/notebook

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


# In[3]:


transactions_train = pd.read_csv('../../../../h-and-m-personalized-fashion-recommendations/transactions_train.csv')
customers_df = pd.read_csv('../../../../h-and-m-personalized-fashion-recommendations/customers.csv')
articles_df = pd.read_csv('../../../../h-and-m-personalized-fashion-recommendations/articles.csv')

class ItemFeatures(ABC):
    @abstractmethod
    def get(self, *args, **kwargs) -> pd.DataFrame:
        """
        article_id -> features
        """
        pass

class CategoryTransform(ItemFeatures):
    """
    factorize all articles columns
    """
    def __init__(self, articles_df: pd.DataFrame):
        self.articles_df = articles_df

    def get(self):
        self.__feature_columns = list(filter(lambda x: 'name' in x, self.articles_df.columns))[1:]
        filtered_articles = self.articles_df[self.__feature_columns]
        filtered_articles = filtered_articles.apply(lambda x: pd.factorize(x)[0])
        filtered_articles['article_id'] = self.articles_df['article_id']

        features = filtered_articles.set_index('article_id').astype('int8')
        return features

    def get_columns(self):
        return self.__feature_columns

class AggrTransform(ItemFeatures):
    """
    aggregation transactions features : mean, max and etc...
    """
    def __init__(self, articles_df: pd.DataFrame, transactions_df: pd.DataFrame):
        self.articles_df = articles_df
        self.transactions_df = transactions_df

    def get(self):
        stats = self._get_stats()
        return stats

    def _get_stats(self):
        transactions_more = self.transactions_df.merge(self.articles_df, on = ('article_id'))
        grouped = (
            transactions_more.
            groupby('article_id')
        )

        counts = (
            grouped['article_id']
            .count()
            .to_frame()
            .rename(columns = {'article_id': 'count'})
            .astype('int16')
            .reset_index()
            .set_index('article_id')
        )
        sums = (
            grouped['price']
            .sum()
            .to_frame()
            .astype('float32')
            .rename(columns = {
                'price': 'sum_price'
            })
        )
        means = (
            grouped['price']
            .mean()
            .to_frame()
            .astype('float32')
            .rename(columns = {
                'price': 'mean_price'
            })
        )
        mins = (
            grouped['price']
            .min()
            .to_frame()
            .astype('float32')
            .rename(columns = {
               'price': 'min_price' 
            })
        )
        maxs = (
            grouped['price']
            .max()
            .to_frame()
            .astype('float32')
            .rename(columns = {
                'price': 'max_price'
            })
        )
        
        output_df = (
            counts
            .merge(sums, on = ('article_id'))
            .merge(means, on = ('article_id'))
            .merge(mins, on = ('article_id'))
            .merge(maxs, on = ('article_id'))
        )
        return output_df

class TopTransforms(ItemFeatures):
    """
    whether category appears in top categories
    """
    def __init__(self, articles_df: pd.DataFrame, topk = 3):
        self.articles_df = articles_df
        self.topk = topk
    
    def get(self):
        name_cols = list(filter(lambda x: 'name' in x, self.articles_df.columns))  
        
        value_counts = self._get_value_counts(name_cols)
        value_counts = {
            f'{k}_{self.topk}': self.articles_df[k].isin(v).astype('int8') for k, v in value_counts.items()
        }
        
        output_df = self.articles_df.assign(**value_counts)
        output_df = output_df[['article_id'] + list(value_counts.keys())].set_index('article_id')
        return output_df
        
    def _get_value_counts(self, name_cols: List[str]):
        value_counts = self.articles_df[name_cols].apply(pd.Series.value_counts)
        get_index = lambda x: value_counts.sort_values(x, ascending = False)[x][:self.topk].index  
        value_counts = dict(zip(name_cols, map(lambda x: get_index(x), name_cols)))
        return value_counts

class ItemFeaturesCollector:
    @staticmethod
    def collect(features: Union[List[ItemFeatures], List[str]], **kwargs) -> pd.DataFrame:
        output_df = None

        for feature in tqdm(features):
            if isinstance(feature, ItemFeatures):
                feature_out = feature.get(**kwargs)
            if isinstance(feature, str):
                try:
                    feature_out = pd.read_csv(feature)
                except:
                    feature_out = pd.read_parquet(feature)

            if output_df is None:
                output_df = feature_out
            else:
                output_df = output_df.merge(feature_out, on = ('article_id'))
        return output_df


# In[ ]:


class UserFeatures(ABC):
    @abstractmethod
    def get(self) -> pd.DataFrame:
        """
        customer_id -> features
        """
        pass

class AggrFeatures(UserFeatures):
    """
    basic aggregation features(min, max, mean and etc...)
    """
    def __init__(self, transactions_df):
        self.groupby_df = transactions_df.groupby('customer_id', as_index = False)

    def get(self):
        output_df = (
            self.groupby_df['price']
            .agg({
                'mean_transactions': 'mean',
                'max_transactions': 'max',
                'min_transactions': 'min',
                'median_transactions': 'median',
                'sum_transactions': 'sum',
                'max_minus_min_transactions': lambda x: x.max()-x.min()
            })
            .set_index('customer_id')
            .astype('float32')
        )
        return output_df

class CountFeatures(UserFeatures):
    """
    basic features connected with transactions
    """
    def __init__(self, transactions_df, topk = 10):
        self.transactions_df = transactions_df
        self.topk = topk

    def get(self):
        grouped = self.transactions_df.groupby('customer_id', as_index = False)
        #number of transactions, number of online articles,
        #number of transactions bigger than mean price of transactions
        a = (
            grouped
            .agg({
                'article_id': 'count',
                'price': lambda x: sum(np.array(x) > x.mean()),
                'sales_channel_id': lambda x: sum(x == 2),
            })
            .rename(columns = {
                'article_id': 'n_transactions',
                'price': 'n_transactions_bigger_mean',
                'sales_channel_id': 'n_online_articles'
            })
            .set_index('customer_id')
            .astype('int8')
        )
        #number of unique articles, number of store articles
        b = (
            grouped
            .agg({
                'article_id': 'nunique',
                'sales_channel_id': lambda x: sum(x == 1),
            })
            .rename(columns = {
                'article_id': 'n_unique_articles',
                'sales_channel_id': 'n_store_articles',
            })
            .set_index('customer_id')
            .astype('int8')
        )
        #number of transactions that are in top
        topk_articles = self.transactions_df['article_id'].value_counts()[:self.topk].index
        c = (
            grouped['article_id']
            .agg({
               f'top_article_{i}':  lambda x: sum(x == k) for i, k in enumerate(topk_articles)
            }
            )
            .set_index('customer_id')
            .astype('int8')
        )
        
        output_df = a.merge(b, on = ('customer_id')).merge(c, on = ('customer_id'))
        return output_df

class CustomerFeatures(UserFeatures):
    """
    All columns from customers dataframe
    """
    def __init__(self, customers_df):
        self.customers_df = self._prepare_customers(customers_df)
    
    def _prepare_customers(self, customers_df):
        customers_df['FN'] = customers_df['FN'].fillna(0).astype('int8')
        customers_df['Active'] = customers_df['Active'].fillna(0).astype('int8')
        customers_df['club_member_status'] = customers_df['club_member_status'].fillna('UNKNOWN')
        customers_df['age'] = customers_df['age'].fillna(customers_df['age'].mean()).astype('int8')
        customers_df['fashion_news_frequency'] = (
            customers_df['fashion_news_frequency']
            .replace('None', 'NONE')
            .replace(np.nan, 'NONE')
        )
        return customers_df

    def get(self):
        output = (
            self.customers_df[filter(lambda x: x != 'postal_code', customers_df.columns)]
            .set_index('customer_id')
        )
        return output

class ArticlesFeatures(UserFeatures):
    """
    returns article features: whether category appears in top categories
    """
    def __init__(self, transactions_df, articles, topk = 10):
        self.merged_df = transactions_df.merge(articles, on = ('article_id'))
        self.articles = articles
        self.topk = topk
    
    def get(self):
        output_df = None

        for col in tqdm(self.articles.columns, desc = 'extracting features'):
            if 'name' in col:
                if output_df is None:
                    output_df = self.aggregate_topk(self.merged_df, col, self.topk)
                else:
                    intermediate_out = self.aggregate_topk(self.merged_df, col, self.topk)
                    output_df = output_df.merge(intermediate_out, on = ('customer_id'))
        return output_df

    def return_value_counts(self, df, column_name, k):
        value_counts = df[column_name].value_counts()[:k].index
        value_counts = list(map(lambda x: x[1], value_counts))
        return value_counts

    def aggregate_topk(self, merged_df, column_name, k):
        grouped_df_indx = merged_df.groupby('customer_id')
        grouped_df = merged_df.groupby('customer_id', as_index = False)
        
        topk_values = self.return_value_counts(grouped_df_indx, column_name, k)
        #how many transactions appears in top category(column)
        n_top_k = (
            grouped_df[column_name]
            .agg({
                f'top_{column_name}_{i}': lambda x: sum(x == k) for i, k in enumerate(topk_values)
            })
            .set_index('customer_id')
            .astype('int16')
        )
        return n_top_k
    
class UserFeaturesCollector:
    """
    collect all features and aggregate them
    """
    @staticmethod
    def collect(features: Union[List[UserFeatures], List[str]], **kwargs) -> pd.DataFrame:
        output_df = None

        for feature in tqdm(features):
            if isinstance(feature, UserFeatures):
                feature_out = feature.get(**kwargs)
            if isinstance(feature, str):
                try:
                    feature_out = pd.read_csv(feature)
                except:
                    feature_out = pd.read_parquet(feature)

            if output_df is None:
                output_df = feature_out
            else:
                output_df = output_df.merge(feature_out, on = ('customer_id'))
        return output_df


# In[ ]:


def preprocFeat():
    item_features = ItemFeaturesCollector.collect([
        CategoryTransform(articles_df),
        AggrTransform(articles_df, transactions_train.iloc[:100_000]),
        TopTransforms(articles_df)])

    item_features.to_parquet('LGBRankerData/itemFeat.parquet')
    
    user_features = UserFeaturesCollector.collect([
        AggrFeatures(transactions_train.iloc[:10_000]),
        CountFeatures(transactions_train.iloc[:10_000], 3),
        CustomerFeatures(customers_df),
        ArticlesFeatures(transactions_train.iloc[:10_000], articles_df, 3),
    ])
    
    user_features.to_parquet('LGBRankerData/userFeat.parquet')

