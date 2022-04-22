#!/usr/bin/env python
# coding: utf-8

# In[143]:


import numpy as np
import pandas as pd
from LGBRanker_Preproc import preprocFeat
from lightgbm import LGBMRanker
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings("ignore")


# In[144]:


def exe_time(func, print_time=True):
    def new_func(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        if print_time:
            print("@%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func


# In[147]:


class LGBMR():
    @exe_time
    def __init__(self, candiN = 15):
        self.candiN = candiN
        preprocFeat()
        self.dataPrep()
    
    @exe_time
    def load(self):
        self.transactions = pd.read_csv('../../../../h-and-m-personalized-fashion-recommendations/transactions_train.csv')
        self.customers = pd.read_parquet('LGBRankerData/userFeat.parquet')
        self.articles = pd.read_parquet('LGBRankerData/itemFeat.parquet')
        self.sub = pd.read_csv('../../../../h-and-m-personalized-fashion-recommendations/sample_submission.csv')
    
    @exe_time
    def preproc(self):
        self.load()
        self.transactions.t_dat = pd.to_datetime(self.transactions.t_dat)    
        self.dfL = []
        for date, i in zip(['2020-08-24', '2020-08-31', '2020-09-07', '2020-09-15'], range(1,5)):
            tmpDf = self.transactions[self.transactions['t_dat'] >= pd.to_datetime(date)]
            self.dfL.append(tmpDf)
            
        self.pcDic0824, self.dummyL0824 = self.preprocA(self.dfL[0])
        self.pcDic0831, self.dummyL0831 = self.preprocA(self.dfL[1])
        self.pcDic0907, self.dummyL0907 = self.preprocA(self.dfL[2])
        self.pcDic0915, self.dummyL0915 = self.preprocA(self.dfL[3])
        
        self.customers[['club_member_status',
                        'fashion_news_frequency']] = (self.customers[['club_member_status',
                                                                      'fashion_news_frequency']]
                                                      .apply(lambda x: pd.factorize(x)[0])).astype('int8')

        self.transactions = self.transactions.merge(self.customers, 
                                                    on = ('customer_id')).merge(self.articles,
                                                                                on = ('article_id'))
        self.transactions.sort_values(['t_dat', 'customer_id'], inplace=True)

        self.train = self.transactions.loc[self.transactions.t_dat <= pd.to_datetime('2020-09-15')].iloc[:1000000]
        self.valid = self.transactions.loc[self.transactions.t_dat >= pd.to_datetime('2020-09-16')]

    def preprocA(self, df):
        pcDic = {}
        for i, x in enumerate(zip(df['customer_id'], df['article_id'])):
            custID, artID = x
            if custID not in pcDic: pcDic[custID] = {}
            if artID not in pcDic[custID]: pcDic[custID][artID] = 0
            pcDic[custID][artID] += 1
        for i in range(1,5):
            dummyL = list((df['article_id'].value_counts()).index)[:12]
        return pcDic, dummyL

    def preprocB(self, custIDL, candiN):
        predDic = {}
        dummyL = list((self.dfL[2]['article_id'].value_counts()).index)[:candiN]
        for i, cid in enumerate(custIDL):
            for pair in [(self.pcDic0915,self.dummyL0915),
                         (self.pcDic0907,self.dummyL0907),
                         (self.pcDic0831,self.dummyL0831),
                         (self.pcDic0824,self.dummyL0824)]:
                pcD = pair[0]
                dL = pair[1]
                if cid in pcD:
                    L = sorted((pcD[cid]).items(), key=lambda x: x[1], reverse=True)
                    L = [var[0] for var in L]
                    if len(L) > candiN:
                        s = L[:candiN]
                        break
                    else:
                        s = L + dL[:(candiN-len(L))]
                        break
                else:
                    s = dummyL
            predDic[cid] = s
        NDf = pd.DataFrame({'customer_id': list(map(lambda x: x[0], predDic.items())), 
                            'negatives': list(map(lambda x: x[1], predDic.items()))})
        NDf = NDf.explode('negatives').rename(columns = {'negatives': 'article_id'})
        return NDf
    
    @exe_time
    def dataPrep(self):
        self.preproc()
        self.train['rank'] = range(len(self.train))
        self.train = self.train.assign(rn = self.train.groupby(['customer_id'])['rank']
                                       .rank(method='first', 
                                             ascending=False)).query("rn <= {}".format(self.candiN))
        self.train = self.train.drop(columns = ['price', 'sales_channel_id', 'rank', 'rn']).sort_values(['t_dat', 'customer_id'])
        self.train['label'] = 1
        
        lastDate = self.train.groupby('customer_id')['t_dat'].max().to_dict()
        neg = self.preprocB(self.train['customer_id'].unique(), self.candiN)
        neg['t_dat'] = neg['customer_id'].map(lastDate)
        neg = neg.merge(self.customers, on = ('customer_id')).merge(self.articles, on = ('article_id'))
        neg['label'] = 0      
        
        self.train = pd.concat([self.train, neg])
        self.train.sort_values(['customer_id', 't_dat'], inplace = True)
        
        self.trainB = self.train.groupby(['customer_id'])['article_id'].count().values
        
    @exe_time
    def LGBMRT(self):
        self.ranker = LGBMRanker(objective="lambdarank",
                                 metric="ndcg",
                                 boosting_type="dart",
                                 max_depth=7,
                                 n_estimators=300,
                                 importance_type='gain', verbose = 10)

        trainX = self.train.drop(columns = ['t_dat', 'customer_id', 'article_id', 'label'])
        trainy = self.train.label
        self.ranker = self.ranker.fit(trainX,
                                      trainy,
                                      group=self.trainB)
    @exe_time 
    def LGBMRP(self, batchS = 1000000):
        test = self.preprocB(self.sub.customer_id.unique(), 12)
        test = test.merge(self.customers, on = ('customer_id')).merge(self.articles, on = ('article_id'))

        pred = []
        for B in tqdm(range(0, len(test), batchS)):
            p = self.ranker.predict(test.iloc[B: B+batchS].drop(columns = ['customer_id', 'article_id']))
            pred.append(p)

        test['pred'] = np.concatenate(pred)
        self.predictions = test[['customer_id', 'article_id', 'pred']]
        self.predictions.sort_values(['customer_id', 'pred'], ascending=False, inplace = True)
        self.predictions = self.predictions.groupby('customer_id')[['article_id']].aggregate(lambda x: x.tolist())
        self.predictions['article_id'] = self.predictions['article_id'].apply(lambda x: ' '.join(['0'+str(k) for k in x]))
        self.predictions = self.sub[['customer_id']].merge(self.predictions.reset_index().rename(columns = {'article_id': 'prediction'}), how = 'left')
        self.predictions['prediction'].fillna(' '.join(['0'+str(d) for d in self.dummyL0907]), inplace = True)
        self.predictions.to_csv('predict_result.csv', index = False)
        return self.predictions

