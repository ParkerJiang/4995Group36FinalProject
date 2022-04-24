#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
from LGBMR_new_Preproc import *
from lightgbm import LGBMRanker
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings("ignore")

def exe_time(func, print_time=True):
    def new_func(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        if print_time:
            print("@%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func

class LGBMR():
    @exe_time
    def __init__(self):
        preprocAll()
        self.preproc()
        
    @exe_time
    def load(self):
        self.transactions = pd.read_parquet('transactions_train.parquet')
        self.customers = pd.read_parquet('customers.parquet')
        self.articles = pd.read_parquet('articles.parquet')
        self.sub = pd.read_csv('../../../../h-and-m-personalized-fashion-recommendations/sample_submission.csv')
    
    @exe_time
    def preproc(self):
        self.load()
        self.getLastP()
        self.getBestS()
        self.concatAll()
    
    @exe_time
    def getLastP(self):
        self.testW = self.transactions.week.max() + 1
        self.transactions = self.transactions[self.transactions.week > self.transactions.week.max() - 10]
        CW = self.transactions.groupby('customer_id')['week'].unique()

        shiftedCW = {}
        for c_id, w in CW.items():
            shiftedCW[c_id] = {}
            for i in range(w.shape[0]-1):
                shiftedCW[c_id][w[i]] = w[i+1]
            shiftedCW[c_id][w[-1]] = self.testW

        self.lastP = self.transactions.copy()
        weeks = []
        for i, (c_id, week) in enumerate(zip(self.transactions['customer_id'], self.transactions['week'])):
            weeks.append(shiftedCW[c_id][week])
        self.lastP.week=weeks
    
    @exe_time
    def getBestS(self):
        meanP = self.transactions.groupby(['week', 'article_id'])['price'].mean()
        sales = self.transactions.groupby('week')['article_id'].value_counts()                 .groupby('week').rank(method='dense', ascending=False)                 .groupby('week').head(12).rename('BSR').astype('int8')
        
        self.BSPreviousW = pd.merge(sales, meanP, on=['week', 'article_id']).reset_index()
        self.BSPreviousW.week += 1

        uniqueT = self.transactions.groupby(['week', 'customer_id']).head(1).drop(columns=['article_id', 'price']).copy()
        self.bestS = pd.merge(uniqueT,
                         self.BSPreviousW,
                         on='week')
        
        testT = uniqueT.drop_duplicates('customer_id').reset_index(drop=True)
        testT.week = self.testW

        self.testBestS = pd.merge(testT,
                             self.BSPreviousW,
                             on='week')

        self.bestS = pd.concat([self.bestS, self.testBestS])
        self.bestS.drop(columns='BSR', inplace=True)
    
    @exe_time
    def concatAll(self):
        self.transactions['purchased'] = 1

        self.fullData = pd.concat([self.transactions, 
                                   self.lastP, 
                                   self.bestS])
        self.fullData.purchased.fillna(0, inplace=True)
        self.fullData.drop_duplicates(['customer_id', 'article_id', 'week'], inplace=True)
        self.fullData = pd.merge(self.fullData,
                                 self.BSPreviousW[['week', 'article_id', 'BSR']],
                                 on=['week', 'article_id'],
                                 how='left')
        
        self.fullData = self.fullData[self.fullData.week != self.fullData.week.min()]
        self.fullData.BSR.fillna(999, inplace=True)
        self.fullData = pd.merge(self.fullData, 
                                 self.articles, 
                                 on='article_id', 
                                 how='left')
        self.fullData = pd.merge(self.fullData, 
                                 self.customers, 
                                 on='customer_id', 
                                 how='left')
        self.fullData.sort_values(['week', 'customer_id'], inplace=True)
        self.fullData.reset_index(drop=True, inplace=True)

        self.train = self.fullData[self.fullData.week != self.testW]
        self.test = self.fullData[self.fullData.week == self.testW]                     .drop_duplicates(['customer_id', 'article_id', 'sales_channel_id']).copy()
        self.trainBaskets = self.train.groupby(['week', 'customer_id'])['article_id'].count().values
        self.feat = ['article_id', 'product_type_no', 'graphical_appearance_no', 
                     'colour_group_code', 'perceived_colour_value_id',
                     'perceived_colour_master_id', 'department_no', 'index_code',
                     'index_group_no', 'section_no', 'garment_group_no', 'FN', 'Active',
                     'club_member_status', 'fashion_news_frequency', 'age', 'postal_code',
                     'BSR']
        self.XTrain = self.train[self.feat]
        self.yTrain = self.train['purchased']
        self.XTest = self.test[self.feat]
        
    @exe_time
    def LGBMRT(self):
        self.ranker = LGBMRanker(objective="lambdarank",
                                 metric="ndcg",
                                 boosting_type="dart",
                                 n_estimators=1,
                                 importance_type='gain', 
                                 verbose = 10)
        
        self.ranker = self.ranker.fit(self.XTrain,
                                      self.yTrain,
                                      group = self.trainBaskets,)

    
    @exe_time 
    def LGBMRP(self, batchS = 1000000):
        
        self.test['predictions'] = self.ranker.predict(self.XTest)
        self.predID = self.test.sort_values(['customer_id', 'predictions'], ascending=False).groupby('customer_id')['article_id'].apply(list).to_dict()
        self.BSLW = self.BSPreviousW[self.BSPreviousW.week == self.BSPreviousW.week.max()]['article_id'].tolist()
        
        self.predictions = []
        def convertInt(S):
            def toInt(S_):
                return int(S_[-16:], 16)
            return S.str[-16:].apply(toInt)
        for c_id in convertInt(self.sub.customer_id):
            pred = self.predID.get(c_id, [])
            pred = pred + self.BSLW
            self.predictions.append(pred[:12])

        self.predictions = [' '.join(['0' + str(p) for p in pred]) for pred in self.predictions]
        self.sub.prediction = self.predictions
        self.sub.to_csv('submission.csv', index=False)




