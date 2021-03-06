# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 18:35:25 2018

@author: Administrator
"""

import codecs
import os
import logging
from collections import OrderedDict,Counter
import numpy as np
import lda
import re
import pickle
log_file="D:/迅雷下载/review_polarity/jst模型DEBUG.log"
logger = logging.getLogger("读取数据DEBUG")
log_level = logging.DEBUG
handler = logging.FileHandler(log_file)
formatter = logging.Formatter("[%(levelname)s][%(funcName)s][%(asctime)s]%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(log_level)
movie_file_path="D:/迅雷下载/review_polarity/txt_sentoken"
stop_path="E:/BaiduYunDownload/stopwords/stop_words_eng.txt"
wordidmap_file="D:/迅雷下载/review_polarity/wordidmap_file.txt"
idwordmap_file="D:/迅雷下载/review_polarity/idwordmap_file.txt"
def load_data(path,sample=False,pos_num=100,neg_num=100):
    dir_list=os.listdir(path)
    pos_path=path+'/'+dir_list[1]
    neg_path=path+'/'+dir_list[0]
    pos_files=os.listdir(pos_path)
    neg_files=os.listdir(neg_path)
    pos=[]
    neg=[]
    for i in pos_files:
        with codecs.open(pos_path+'/'+i,'r','utf-8') as f:
            temp=''
            for line in f.readlines():
                temp+=line
        temp = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", " ",temp)
        if temp not in pos:
            pos.append(temp)                
        logger.info("%s 正情感已读取完毕"%(i))
        if sample and len(pos)==pos_num:
            break
    for i in neg_files:
        with codecs.open(neg_path+'/'+i,'r','utf-8') as f:
            temp=''
            for line in f.readlines():
                temp+=line
        if temp not in neg:
            neg.append(temp)         
        temp = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", " ",temp)
        logger.info("%s 负情感已读取完毕"%(i))
        if sample and len(neg)==neg_num:
            break
    print("共有%d个正面文档，%d个负面文档"%(len(pos),len(neg)))
    return pos,neg
def fenci(text):
    from nltk.tokenize import WordPunctTokenizer
    return WordPunctTokenizer().tokenize(text)
def cleanwords(words):
    clean_words=[]
    stop=[]
    with codecs.open(stop_path,'r') as f:
        for line in f.readlines():
            stop.append(line.strip().split())
    stop=sum(stop,[])
    stopwords={}.fromkeys(stop)
    for word in words:
        if word not in stopwords and len(word)>=3:
            clean_words.append(word)
    return clean_words
class Article(object):
    def __init__(self):
        self.words=[]
        self.length=0
class Documents(object):
    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        self.docs = []
        self.word2id = OrderedDict()
        self.id2word=OrderedDict()
    def cachewordidmap(self,word2id_file,id2word_file):
        with codecs.open(word2id_file, 'w') as f:
            for word,word_id in self.word2id.items():
                f.write(word +"\t"+str(id)+"\n")
        with codecs.open(id2word_file, 'w') as f:
            for word_id,word in self.id2word.items():
                f.write(str(word_id) +"\t"+word+"\n")
def preprocess(words_list,word2id_file,id2word_file):
        docs=Documents()
        words_idx=0
        for review in words_list:
            article=Article()
            for word in review:
                if word in docs.word2id:
                    article.words.append(docs.word2id[word])
                else:
                    docs.word2id[word] = words_idx
                    docs.id2word[words_idx]=word
                    article.words.append(words_idx)
                    words_idx += 1
            article.length=len(review)
            docs.docs.append(article)
        docs.docs_count = len(docs.docs)
        docs.words_count = len(docs.word2id)
        logger.info(u"共有%s个文档" % docs.docs_count)
        docs.cachewordidmap(word2id_file,id2word_file)
        logger.info(u"词与序号对应关系已保存到%s" % word2id_file)
        logger.info(u"序号与词对应关系已保存到%s" % id2word_file)
        return docs
def jst_showtopN_words(id2word,phi_matrix,topN):
    topwords_list=[]
    for i in range(phi_matrix.shape[1]):
        temp=[]
        for j in range(phi_matrix.shape[2]):
            keys=np.argsort(phi_matrix[:,i,j])[-topN:]
            words=list(map(id2word.get,keys))
            temp.append(words)
        topwords_list.append(temp)
    return topwords_list
def main():
    pos,neg=load_data(movie_file_path,True,10,10)
    total_words=list(map(cleanwords,map(fenci,pos+neg)))
    total_words=Stem_all(total_words)
    total_words=list(map(cleanwords,total_words))
    a=preprocess(total_words,wordidmap_file,idwordmap_file)
    jst=JST_model(a,K=30,S=2,alpha=50/30,gamma=0.01,iterations=1000)
    jst.Gibbs_sampling()
    top=jst_showtopN_words(a.id2word,jst.phi,20)
    return jst,jst.theta,jst.phi,jst.pai,top
def Stem(words_list):
    import nltk.stem
    s=nltk.stem.SnowballStemmer("english")    
    return list(map(s.stem,words_list))
def Stem_all(corpus):
    return list(map(Stem,corpus))
class JST_model(object):
    def __init__(self,dpre,seed=1,K=10,S=2,alpha=1,beta=0.01,gamma=1,iterations=1000,topN_words=5):
        '''
        先初始化超参数K（主题数），两个狄利克雷分布的参数alpha，beta
        '''
    
        self.dpre = dpre #获取预处理参数
        self.K=K
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.V=dpre.words_count
        self.S=S
        self.phi=np.zeros((self.V,self.K,self.S))
        self.theta=np.zeros((self.dpre.docs_count,self.K,self.S))
        self.pai=np.zeros((self.dpre.docs_count,self.S))
        self.iter=iterations
        self.top_words_num=topN_words
        
#        self.M=dpre.doc_count
        # p,概率向量 double类型，存储采样的临时变量
        # nw,词word在主题topic上的分布
        # nwsum,每各topic的词的总数
        # nd,每个doc中各个topic的词的总数
        # ndsum,每各doc中词的总数
        self.p=np.zeros((self.K,self.S))
#        self.p_sentiment=np.zeros(self.S)

        self.nstw = np.zeros((self.V,self.K,self.S),dtype="int")#词i被分配到情感j,主题k的数量    
        self.nstwsum = np.zeros((self.K,self.S),dtype="int")    #被分配到情感j，主题k的词数量
        self.nstd = np.zeros((dpre.docs_count,self.K,self.S),dtype="int")  #第i篇文档中被分配到情感j，主题k的词数量     
        self.nsdsum = np.zeros((dpre.docs_count,self.S),dtype="int")    #第i篇文档情感j的数量
        self.ndsum = np.zeros(dpre.docs_count,dtype="int")  #每篇文档词数量
        np.random.seed(seed)
        self.Z = np.array([ [[np.random.randint(0,self.K),np.random.randint(0,self.S)] \
                         for y in range(dpre.docs[x].length)] for x in range(dpre.docs_count)])        # M*doc.size()，文档中词的主题分布
        for doc in range(len(self.Z)):
            for word in range(len(self.Z[doc])):
                topic_index=self.Z[doc][word][0]
#                self.Z[doc][word][0]=topic_index
                sentiment_index= self.Z[doc][word][1]
#                self.Z[doc][word][1]=sentiment_index
                self.nstw[self.dpre.docs[doc].words[word]][topic_index][sentiment_index]+=1
                self.nstwsum[topic_index][sentiment_index]+=1
                self.nstd[doc][topic_index][sentiment_index]+=1
                self.nsdsum[doc][sentiment_index]+=1
                self.ndsum[doc]+=1
    def Gibbs_sampling(self):
    #,thetafile,phifile,parafile,topNfile,tassginfile
        for i in range(self.iter):
            for m in range(self.dpre.docs_count):
                for word in range(len(self.Z[m])):
                    topic,sentiment=self.Z[m][word]
                    word_id=self.dpre.docs[m].words[word]
                    self.nstw[word_id][topic][sentiment]-=1
                    self.nstd[m][topic][sentiment]-=1
                    self.nstwsum[topic][sentiment]-=1
                    self.nsdsum[m][sentiment]-=1
                    self.ndsum[m]-=1
                    self.p=(self.nstd[m]+self.alpha)/(self.nsdsum[m]+self.K*self.alpha)\
                       *(self.nstw[word_id]+self.beta)/(self.nstwsum+self.V*self.beta)\
                       *(self.nsdsum[m]+self.gamma)/(self.ndsum[m]+self.S*self.gamma)
                    prob=self.p.flatten()
                    assert(len(prob)==self.K*self.S)
                    for i in range(1,len(prob)):
                        prob[i]+=prob[i-1]
#                    print(prob[-11])
#                    prob=prob/np.sum(prob)
#                    u=np.random.rand()
                    u=np.random.uniform(0.0,prob[-1])
                    for senti_top in range(len(prob)):
                        if prob[senti_top]>u:
                            break
                    if senti_top%self.S:
                        new_sentiment=senti_top%self.S-1
                        new_topic=senti_top//self.S
                    else:
                        new_sentiment=self.S-1
                        new_topic=senti_top//self.S-1
#                    if senti_top//self.S:
#                        new_topic=senti_top//self.S
#                    else:
#                        new_topic=senti_top//self.S-1
                    self.nstw[word_id][new_topic][new_sentiment]+=1
                    self.nstwsum[new_topic][new_sentiment]+=1
                    self.nstd[m][new_topic][new_sentiment]+=1
                    self.nsdsum[m][new_sentiment]+=1
                    self.ndsum[m]+=1
                    self.Z[m][word]=[new_topic,new_sentiment]
        logger.info(u"迭代完成。")
        
        logger.debug(u"计算文章-主题分布")
        self._theta()
        logger.debug(u"计算词-主题分布")
        self._phi()
        logger.debug("计算文档-情感分布")
        self._pai()
#        logger.debug(u"保存模型")
#        self.save(thetafile,phifile,parafile,topNfile,tassginfile)
    def _theta(self):
        for i in range(self.dpre.docs_count):
            self.theta[i]=(self.nstd[i]+self.alpha)/(self.nsdsum[i]+self.K*self.alpha)
    def _phi(self):
        for i in range(self.V):
            self.phi[i]=(self.nstw[i]+self.beta)/(self.nstwsum+self.V*self.beta)
    def _pai(self):
        for i in range(self.dpre.docs_count):
            self.pai[i]=(self.nsdsum[i]+self.gamma)/(self.ndsum[i]+self.S*self.gamma)    


    
    