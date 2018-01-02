# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 11:15:13 2017

@author: Administrator
"""

import codecs
import os
import logging
import jieba
from collections import OrderedDict,Counter
import numpy as np
import lda
import re
log_file="E:/文本特征提取/谭松波--酒店评论语料/谭松波--酒店评论语料/utf-8/读取数据DEBUG.log"
logger = logging.getLogger("读取数据DEBUG")
log_level = logging.DEBUG
handler = logging.FileHandler(log_file)
formatter = logging.Formatter("[%(levelname)s][%(funcName)s][%(asctime)s]%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(log_level)
file_path="E:/文本特征提取/谭松波--酒店评论语料/谭松波--酒店评论语料/utf-8/2000/2000"
stop_path="E:/文本特征提取/谭松波--酒店评论语料/stopwords.txt"
wordidmap_file="E:/文本特征提取/谭松波--酒店评论语料/wordidmap_file.txt"
idwordmap_file="E:/文本特征提取/谭松波--酒店评论语料/idwordmap_file.txt"
def load_data(path):
    dir_list=os.listdir(path)
    pos_path=path+'/'+dir_list[1]
    neg_path=path+'/'+dir_list[0]
    pos_files=os.listdir(pos_path)
    neg_files=os.listdir(neg_path)
    pos=[]
    neg=[]
    reg=re.compile("[^x00-xff|\u3000]")
    for i in pos_files:
        temp=[]
        with codecs.open(pos_path+'/'+i,'r','utf-8') as f:
            for line in f.readlines():
#                line=re.sub("[A-Za-z0-9\!\%\[\]\,\。\？\、\……\?]", "", line).strip()
                line=("".join(reg.findall(line))).strip()
                if line !='' and line[:4]!="宾馆反馈" and line not in temp:
                    temp.append(line)
        if temp not in pos:
            pos.append(temp)                
        logger.info("%s 正情感已读取完毕"%(i))
    for i in neg_files:
        temp=[]
        with codecs.open(neg_path+'/'+i,'r','utf-8') as f:
            for line in f.readlines():
                line=("".join(reg.findall(line))).strip()
                if line !='' and line[:4]!="宾馆反馈" :
                    temp.append(line)
        if temp not in neg:
            neg.append(temp)                
        logger.info("%s 负情感已读取完毕"%(i))
    return pos,neg
def fenci(docs,stop_path):
    stop_words=[]
    words=[]
    jieba.load_userdict("E:/文本特征提取/谭松波--酒店评论语料/user_dict.txt")
    with codecs.open(stop_path,'r','utf-8') as f:
        for line in f.readlines():    
            stop_words.append(line.strip())
    for review in docs:
        temp=[]
        for sentence in review:
            cut_word=jieba.cut(sentence.strip())
            for word in cut_word:
                if word not in stop_words and word !=" ":
                    temp.append(word)
        if len(temp)>0:
            words.append(temp)
    return words
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
        with codecs.open(word2id_file, 'w','utf-8') as f:
            for word,word_id in self.word2id.items():
                f.write(word +"\t"+str(id)+"\n")
        with codecs.open(id2word_file, 'w','utf-8') as f:
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
                    prob=prob/np.sum(prob)
                    u=np.random.rand()
                    for senti_top in range(len(prob)):
                        if prob[senti_top]>u:
                            break
                    new_sentiment=senti_top%self.S
                    new_topic=senti_top//self.S
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
#    def save(self):
        
def list2matrix(words_list):
    word_Matrix=[]
    total_words=sum(words_list,[])
#    words_num=len(total_words)
    words_distinct=list(set(total_words))
    for review in words_list:
        review_dic=Counter(review)
        word_key=list(review_dic.keys())
        row=[]
        for i in range(len(words_distinct)):
            if words_distinct[i] in word_key:
                row.append(review_dic[words_distinct[i]])
            else:
                row.append(0)
        word_Matrix.append(row)
    return np.array(word_Matrix),words_distinct
def train_test_split(data,seed=2,train_rate=0.8):
    '''
    data的格式为np.array
    '''
    data_all=np.array(data)
    np.random.seed(seed)
    np.random.shuffle(data_all)
    pos_index=np.where(data_all[:,-1]==1)[0]
    neg_index=np.where(data_all[:,-1]==-1)[0]
    train_pos=data_all[pos_index[:int(train_rate*len(pos_index))],:]
    test_pos=data_all[pos_index[int(train_rate*len(pos_index)):],:]
    train_neg=data_all[neg_index[:int(train_rate*len(neg_index))],:]
    test_neg=data_all[neg_index[int(train_rate*len(neg_index)):],:]
    train=np.concatenate((train_pos,train_neg),axis=0)
    test=np.concatenate((test_pos,test_neg),axis=0)
    return train,test
    
    
pos,neg=load_data(file_path)
#del pos[5039]
pos_words=fenci(pos,stop_path)
y_pos=np.ones(len(pos_words))
neg_words=fenci(neg,stop_path)
y_neg=-np.ones(len(neg_words))
y=np.concatenate((y_pos,y_neg),axis=0)
y=y.reshape((-1,1))
pos_words.extend(neg_words)
total_words=pos_words
total_matrix,vacabulary=list2matrix(total_words)

data_all=np.concatenate((total_matrix,y),axis=1)
def train_with_lda(data_matrix,split=False,topic_num=5,alpha=0.1,beta=0.01,iterations=1000):
    if split:
        train,test=train_test_split(data_matrix)
    else:
        train=data_matrix
    lda_model=lda.LDA(5,1000)
    lda_model.fit(train[:,:-1])
    topic_word=lda_model.topic_word_
    doc_topic_train=lda_model.doc_topic_
    if split:
        doc_topic_pred=lda_model.transform(test[:,:-1])
    else:
        doc_topic_pred=lda_model.doc_topic_
    
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier(100)
    clf.fit(doc_topic_train,train[:,-1])
    pred=clf.predict(doc_topic_pred[:,:-1])
    accuracy=sum(pred==test[:,-1])/len(pred)
    return accuracy,topic_word

def lda_once(data,topic_num,alpha=0.1,beta=0.01,iterations=1000):
    lda_model=lda.LDA(topic_num,iterations,alpha,beta)
    lda_model.fit(data.astype(int))
    return lda_model.topic_word_,lda_model.doc_topic_
def show_topic_words(vacabulary,topic_word,topN=10):
    topN_words=[]
    V=np.array(vacabulary)
    for i in topic_word.T:
        index=list(reversed(np.argsort(i)[-topN:]))
        topN_words.append(V[index])
    return topN_words
def search_word(word,words_list):
    index=[]
    for i in range(len(words_list)):
        if word in words_list[i]:
            index.append(i)
    return index
doc_topic,topic_word=lda_once(data_all[:,:-1],5)
topN=show_topic_words(vacabulary,topic_word,20)

#clf=RandomForestClassifier(100)
#clf.fit(doc_topic,y)
#for i in range(100):
#    t1,t2=train_test_split(data_all,seed=i)
#    if np.sum(t1,axis=0).min()>0 and np.sum(t1,axis=1).min()>0:
#        print(i)
phi[1][2][:5]