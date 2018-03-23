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
import pickle
import time
import matplotlib.pyplot as plt
log_file="E:/文本特征提取/谭松波--酒店评论语料/谭松波--酒店评论语料/utf-8/jst模型DEBUG.log"
logger = logging.getLogger("读取数据DEBUG")
log_level = logging.DEBUG
handler = logging.FileHandler(log_file)
formatter = logging.Formatter("[%(levelname)s][%(funcName)s][%(asctime)s]%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(log_level)
file_path="E:/文本特征提取/谭松波--酒店评论语料/谭松波--酒店评论语料/utf-8/2000/2000"
movie_file_path="D:/迅雷下载/review_polarity/txt_sentoken"
stop_path="E:/文本特征提取/谭松波--酒店评论语料/stopwords.txt"
wordidmap_file="E:/文本特征提取/谭松波--酒店评论语料/wordidmap_file.txt"
idwordmap_file="E:/文本特征提取/谭松波--酒店评论语料/idwordmap_file.txt"
wordidmap_file="E:/文本特征提取/谭松波--酒店评论语料/wordidmap_file.txt"
idwordmap_file="E:/文本特征提取/谭松波--酒店评论语料/idwordmap_file.txt"
seed_path="C:/Users/Administrator/Downloads/SentimentAnalysisWordsLib (1)/SentimentAnalysisWordsLib/清华大学李军中文褒贬义词典"
word2idmap_sentence="E:/文本特征提取/谭松波--酒店评论语料/word2idmapsentence.txt"
id2wordmap_sentence="E:/文本特征提取/谭松波--酒店评论语料/id2wordmapsentence.txt"
def load_data(path,sample=False,pos_num=100,neg_num=100):
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
        if sample and len(pos)==pos_num:
            break
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
        if sample and len(neg)==neg_num:
            break
    print("共有%d个正面文档，%d个负面文档"%(len(pos),len(neg)))
    return pos,neg,len(pos),len(neg)
def fenci(docs,stop_path,freq=0):
    stop_words=[]
    words=[]
    jieba.load_userdict("E:/文本特征提取/谭松波--酒店评论语料/user_dict.txt")
    with codecs.open(stop_path,'r','utf-8') as f:
        for line in f.readlines():    
            stop_words.append(line.strip())
    word_dict={}
    for review in docs:
        temp=[]
        for sentence in review:
            cut_word=jieba.cut(sentence.strip())
            for word in cut_word:
                if word not in stop_words and word !=" ":
                    if word in word_dict:
                        word_dict[word]+=1
                    else:
                        word_dict[word]=1
                    temp.append(word)
        if len(temp)>0:
            words.append(temp)
    count=Counter(word_dict).most_common()
    words_save={}.fromkeys([word for word,num in count if num>=freq])
    words_remain=[]
    for review in words:
        temp=[]
        for word in review:
            if word in words_save:
                temp.append(word)
        if len(temp)>0:
            words_remain.append(temp)
    return words_remain
def seed_words(seed_path):
    dir_list=os.listdir(seed_path)
    neg_seed=[]
    pos_seed=[]
    with codecs.open(seed_path+'/'+dir_list[0],'r','utf-8') as f:
        for line in f.readlines():
            neg_seed.append(line.strip())
    with codecs.open(seed_path+'/'+dir_list[1],'r','utf-8') as f:
        for line in f.readlines():
            pos_seed.append(line.strip())
    neg_seed={}.fromkeys(neg_seed)
    pos_seed={}.fromkeys(pos_seed)
    return neg_seed,pos_seed    
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
    def __init__(self,dpre,seed=1,K=10,S=2,alpha=1,beta=0.01,gamma=[1,1],iterations=1000,topN_words=5):
        '''
        先初始化超参数K（主题数），两个狄利克雷分布的参数alpha，beta
        '''
    
        self.dpre = dpre #获取预处理参数
        self.K=K
        self.alpha=alpha
        self.beta=beta
        self.gamma=np.array(gamma)
        self.V=dpre.words_count
        self.S=S
        assert(len(self.gamma)==self.S)
        self.phi=np.zeros((self.V,self.K,self.S))
        self.theta=np.zeros((self.dpre.docs_count,self.K,self.S))
        self.pai=np.zeros((self.dpre.docs_count,self.S))
        self.iter=iterations
        self.top_words_num=topN_words
        self.p=np.zeros((self.K,self.S))
        self.nstw = np.zeros((self.V,self.K,self.S),dtype="int")#词i被分配到情感j,主题k的数量    
        self.nstwsum = np.zeros((self.K,self.S),dtype="int")    #被分配到情感j，主题k的词数量
        self.nstd = np.zeros((dpre.docs_count,self.K,self.S),dtype="int")  #第i篇文档中被分配到情感j，主题k的词数量     
        self.nsdsum = np.zeros((dpre.docs_count,self.S),dtype="int")    #第i篇文档情感j的数量
        self.ndsum = np.zeros(dpre.docs_count,dtype="int")  #每篇文档词数量
        np.random.seed(seed)
        self.initialize_Z(seed_path)
        # M*doc.size()，文档中词的主题分布
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
    def initialize_Z(self,seed_path):
        neg_seed,pos_seed=seed_words(seed_path)
        self.Z = np.array([ [[np.random.randint(0,self.K),self.initial_S(self.dpre.id2word[self.dpre.docs[x].words[y]],neg_seed,pos_seed)] \
                         for y in range(self.dpre.docs[x].length)] for x in range(self.dpre.docs_count)])
    def initial_S(self,word,neg,pos):
        if word in neg:
            return 1
        elif word in pos:
            return 0
        else:
            return np.random.randint(0,self.S)
    def Gibbs_sampling(self):
    #,thetafile,phifile,parafile,topNfile,tassginfile
        np.random.seed(1)
        print("训练开始")
        for i in range(self.iter):
            if i%100==0 and i>0:
                print("完成%d次迭代"%i)
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
                       *(self.nsdsum[m]+self.gamma)/(self.ndsum[m]+np.sum(self.gamma))
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
                  
                    new_sentiment=senti_top%self.S
                    new_topic=senti_top//self.S

                    self.nstw[word_id][new_topic][new_sentiment]+=1
                    self.nstwsum[new_topic][new_sentiment]+=1
                    self.nstd[m][new_topic][new_sentiment]+=1
                    self.nsdsum[m][new_sentiment]+=1
                    self.ndsum[m]+=1
                    self.Z[m][word]=[new_topic,new_sentiment]
        logger.info(u"迭代完成。")
        print("训练结束")
        
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
    
    
#pos,neg=load_data(file_path,True,10,10)
##del pos[5039]
#pos_words=fenci(pos,stop_path)
#y_pos=np.ones(len(pos_words))
#neg_words=fenci(neg,stop_path)
#y_neg=-np.ones(len(neg_words))
#y=np.concatenate((y_pos,y_neg),axis=0)
#y=y.reshape((-1,1))
#pos_words.extend(neg_words)
#total_words=pos_words
#total_matrix,vacabulary=list2matrix(total_words)
#
#data_all=np.concatenate((total_matrix,y),axis=1)
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
    for i in topic_word:
        index=list(reversed(np.argsort(i)[-topN:]))
        topN_words.append(V[index])
    return topN_words
def search_word(word,words_list):
    index=[]
    for i in range(len(words_list)):
        if word in words_list[i]:
            index.append(i)
    return index
def Sentiment_seed(sentiment_path):
    senti_seed=[]
    with codecs.open(sentiment_path,'r','utf-8') as f:
        for line in f.readlines():
            senti_seed.append(line.strip())
    return {}.fromkeys(senti_seed)
def cut_sentence(paragraph):
    cut=re.split(r"[；。，！……？;.,?!]",paragraph)[:-1]
    temp=[]
    for i in cut:
        if len(i)>0:
            temp.append(i)
    return temp
def fenci_paragraph(paragraph,stop=stop_path):
    stop_words=[]
    words=[]
    jieba.load_userdict("E:/文本特征提取/谭松波--酒店评论语料/user_dict.txt")
    with codecs.open(stop,'r','utf-8') as f:
        for line in f.readlines():    
            stop_words.append(line.strip())
    for sentence in paragraph:
        temp=[]
        cut_word=jieba.cut(sentence.strip())
        for word in cut_word:
            if word not in stop_words and word !=' ':
                temp.append(word)
        if len(temp)>0:
            words.append(temp)
    return words
def combine(corpus):
    temp=[]
    for doc in corpus:
        if len(doc)>1:
            temp.append(sum(doc,[]))
        else:
            temp.append(doc[0])
    return temp
def load_sentence(data_path,sample=False,pos_num=100,neg_num=100):
    if sample:
        pos,neg,pos_num,neg_num=load_data(data_path,sample,pos_num,neg_num)
    else:
        pos,neg,pos_num,neg_num=load_data(data_path)
    corpus=pos+neg
    corpus_sentence=[]
    for review in corpus:
        corpus_sentence.append(list(map(cut_sentence,review)))
    return corpus_sentence,pos_num,neg_num

def filter_words(words_list,freq=0):
    word_dict={}
    temp=sum(sum(words_list,[]),[])
    for word in temp:
        if word in word_dict:
            word_dict[word]+=1
        else:
            word_dict[word]=1
    
    count=Counter(word_dict).most_common()
    words_save={}.fromkeys([word for word,num in count if num>=freq])
    words_remain=[]
    for review in words_list:
        article=[]
        for sentence in review:
            senten=[]
            for word in sentence: 
                if word in words_save:
                    senten.append(word)
            if len(senten)>0:
                article.append(senten)
        if len(article)>0:
            words_remain.append(article)
    return words_remain
class Article2():
    def __init__(self):
        self.sentence=0
        self.words=[]
        self.each_sentence_length=[]
class Document2():
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
def preprocess2(words_list,word2id_file,id2word_file):
        docs=Document2()
        words_idx=0
        for review in words_list:
            article=Article2()
            for sentence in review:
                temp=[]
                for word in sentence:
                    if word in docs.word2id:
                        temp.append(docs.word2id[word])
                    else:
                        docs.word2id[word] = words_idx
                        docs.id2word[words_idx]=word
                        temp.append(words_idx)
                        words_idx += 1
                article.each_sentence_length.append(len(temp))
                article.words.append(temp)
            article.sentence=len(article.words)
            docs.docs.append(article)
        docs.docs_count = len(docs.docs)
        docs.words_count = len(docs.word2id)
        logger.info(u"共有%s个文档" % docs.docs_count)
        docs.cachewordidmap(word2id_file,id2word_file)
        logger.info(u"词与序号对应关系已保存到%s" % word2id_file)
        logger.info(u"序号与词对应关系已保存到%s" % id2word_file)
        return docs
def factor(a,delta):
    temp=1
    for j in range(delta):
        temp*=(a+j)
    return temp
 
def factor_vectorize(vector,delta_list):
    temp=np.ones(vector.shape)
    for i in range(vector.shape[-1]):
        for j in range(delta_list[i]):
            temp[:,:,i]*=(vector[:,:,i]+j)
    return temp
    
#    return list(map(factor,vector[:,:],delta_list))
class ASUM_model():
    def __init__(self,dpre,initialize=True,K=5,S=2,seed=1,iterations=1000,alpha=1,beta=0.01,gamma=1):
        self.dpre=dpre
        self.K=K
        self.S=S
        self.iter=iterations
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.V=dpre.words_count
        self.phi=np.zeros((self.V,self.K,self.S))
        self.theta=np.zeros((self.dpre.docs_count,self.K,self.S))
        self.pai=np.zeros((self.dpre.docs_count,self.S))
        self.p=np.zeros((self.K,self.S))
        self.nstw = np.zeros((self.K,self.S,self.V),dtype="float")#词i被分配到情感j,主题k的数量    
        self.nstwsum = np.zeros((self.K,self.S),dtype="float")    #被分配到情感j，主题k的词数量
        self.nstd = np.zeros((dpre.docs_count,self.K,self.S),dtype="float")  #第i篇文档中被分配到情感j，主题k的词数量     
        self.nsdsum = np.zeros((dpre.docs_count,self.S),dtype="float")    #每篇文档的不同情感句子数量
        self.nstdsum=np.zeros((self.dpre.docs_count,self.K,self.S))
        self.ndsum = np.zeros(dpre.docs_count,dtype="float")  #每篇文档句子数量
        np.random.seed(seed)
        if initialize:
            self.initialize_Z(seed_path)
        else:
            self.Z = np.array([ [[np.random.randint(0,self.K),np.random.randint(0,self.S),self.dpre.docs[x].words[y]] \
                         for y in range(self.dpre.docs[x].sentence)] for x in range(self.dpre.docs_count)]) 
        for doc in range(len(self.Z)):
            for sentence in range(len(self.Z[doc])):
                topic_index=self.Z[doc][sentence][0]
#                self.Z[doc][word][0]=topic_index
                sentiment_index= self.Z[doc][sentence][1]
                temp=Counter(self.Z[doc][sentence][2])
                words_id=list(temp.keys())
                words_count=np.array(list(temp.values()))
                words_num=self.dpre.docs[doc].each_sentence_length[sentence]
                self.nstw[topic_index,sentiment_index,words_id]+=words_count
                self.nstwsum[topic_index][sentiment_index]+=words_num
                self.nstd[doc][topic_index][sentiment_index]+=1
                self.nsdsum[doc][sentiment_index]+=1
                self.ndsum[doc]+=1
    def Gibbs_sampling(self):
        np.random.seed(1)
        print("训练开始")
        for i in range(self.iter):
            if i %100==0 and i>0:
                print("已完成%d次迭代"%i)
            for m in range(self.dpre.docs_count):
                for sentence in range(len(self.Z[m])):
                    topic,sentiment=self.Z[m][sentence][:2]
#                    word_id=self.dpre.docs[m].words[word]
                    temp=Counter(self.Z[m][sentence][2])
                    words_id=list(temp.keys())
                    words_count=np.array(list(temp.values()))
                    words_num=self.dpre.docs[m].each_sentence_length[sentence]
                    self.nstw[topic,sentiment,words_id]-=words_count
                    self.nstd[m][topic][sentiment]-=1
                    self.nstwsum[topic][sentiment]-=words_num
                    self.nsdsum[m][sentiment]-=1
                    self.ndsum[m]-=1
                    self.p=(self.nstd[m]+self.alpha)/(self.nsdsum[m]+self.K*self.alpha)\
                       *np.prod(factor_vectorize(self.nstw[:,:,words_id]+self.beta,words_count),2)/factor(self.nstwsum+self.V*self.beta,words_num)\
                       *(self.nsdsum[m]+self.gamma)/(self.ndsum[m]+self.gamma*self.S)
                    prob=self.p.flatten()
                    assert(len(prob)==self.K*self.S)
                    assert((prob>0).all() and (prob!=np.inf).any())
                    for i in range(1,len(prob)):
                        prob[i]+=prob[i-1]
#                    print(prob[-1])
                    u=np.random.uniform(0.0,prob[-1])
                    for senti_top in range(len(prob)):
                        if prob[senti_top]>u:
                            break
                    new_sentiment=senti_top%self.S
                    new_topic=senti_top//self.S
                    self.nstw[new_topic,new_sentiment,words_id]+=words_count
                    self.nstwsum[new_topic][new_sentiment]+=words_num
                    self.nstd[m][new_topic][new_sentiment]+=1
                    self.nsdsum[m][new_sentiment]+=1
                    self.ndsum[m]+=1
                    self.Z[m][sentence][:2]=[new_topic,new_sentiment]
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
            self.phi[i]=(self.nstw[:,:,i]+self.beta)/(self.nstwsum+self.V*self.beta)
    def _pai(self):
        for i in range(self.dpre.docs_count):
            self.pai[i]=(self.nsdsum[i]+self.gamma)/(self.ndsum[i]+self.S*self.gamma)
    def initialize_Z(self,seed_path):
        neg_seed,pos_seed=seed_words(seed_path)
        self.Z = np.array([ [[np.random.randint(0,self.K),self.initial_S(self.dpre.docs[x].words[y],neg_seed,pos_seed),self.dpre.docs[x].words[y]] \
                         for y in range(self.dpre.docs[x].sentence)] for x in range(self.dpre.docs_count)],dtype=object)
        
    def initial_S(self,wordsid_list,neg,pos):
        pos_num=0
        neg_num=0
        words_list=map(self.dpre.id2word.get,wordsid_list)
        for word in words_list:
            if word in neg:
                neg_num+=1
            elif word in pos:
                pos_num+=1
        if pos_num>neg_num:
            return 0
        elif neg_num>pos_num:
            return 1
        else:
            return np.random.randint(0,self.S)
    #doc_topic,topic_word=lda_once(data_all[:,:-1],5)
#topN=show_topic_words(vacabulary,topic_word,20)

#clf=RandomForestClassifier(100)
#clf.fit(doc_topic,y)
#for i in range(100):
#    t1,t2=train_test_split(data_all,seed=i)
#    if np.sum(t1,axis=0).min()>0 and np.sum(t1,axis=1).min()>0:
#        print(i)
#a=preprocess(total_words,wordidmap_file,idwordmap_file)
#jst=JST_model(a,iterations=1000,S=2)
#jst.Gibbs_sampling()
#theta,phi,pai=jst.theta,jst.phi,jst.pai
#with open("E:/硕士毕业论文/jst_with2S.pkl",'wb') as f:
#    pickle.dump(jst,f)
def showtopN_words(id2word,phi_matrix,topN):
    topwords_list=[]
    for i in range(phi_matrix.shape[1]):
        temp=[]
        for j in range(phi_matrix.shape[2]):
            keys=np.argsort(phi_matrix[:,i,j])[-topN:]
            words=list(map(id2word.get,keys))
            temp.append(words)
        topwords_list.append(temp)
    return topwords_list
def lookintofiles(filepath,stoppath):
    pos,neg=load_data(filepath)
    pos_words=fenci(pos,stoppath)
    neg_words=fenci(neg,stoppath)
    pos_words.extend(neg_words)
#    total_words=pos_words
    return pos,neg,pos_words,neg_words 

  
def main():
    topwords_path="E:/文本特征提取/谭松波--酒店评论语料/top_words.txt"
    pos,neg,pos_num,neg_num=load_data(file_path,True,10,10)
    corpus=pos+neg
    corpus_words=fenci(corpus,stop_path,5)
    a=preprocess(corpus_words,wordidmap_file,idwordmap_file)
    result_pai=[]
    result_phi=[]
    result_theta=[]
    result_top=[]
    for i in [5,10]:
        
        jst=JST_model(a,K=i,alpha=50/i,iterations=1000,gamma=[0.01,0.9],S=2)
        jst.Gibbs_sampling()
        print("主题数为%d训练完毕"%(i))            
        top=showtopN_words(a.id2word,jst.phi,10)
        calculate_accuracy(jst.pai,pos_num,neg_num)
        with codecs.open(topwords_path,'a','utf-8') as f:
            f.write("===================================================\n")
            f.write("主题数为%d\n" %i)
            for i in range(len(top)):
                f.write("主题编号%d:\n"%i)
                for j in range(len(top[i])):
                    f.write("情感编号为%d:"%j)
                    for word in top[i][j]:
                        f.write(word)
                        f.write("\t")
                    f.write("\n")
            f.write("===================================================\n")
                
        result_theta.append((i,jst.theta))
        result_phi.append((i,jst.phi))
        result_pai.append((i,jst.pai))
        result_top.append((i,top))
#    with open("E:/硕士毕业论文/jst_with"+str(jst.S)+"S第"+str(i)+"次实验数据为"+file_path[-4:]+'共'+str(len(pos)+len(neg))+"篇文档.pkl",'wb') as f:
#        pickle.dump(jst,f)
#    return jst,jst.theta,jst.phi,jst.pai,top
    return result_theta,result_phi,result_pai,result_top
#jst,theta,phi,pai,top=main(17)
#theta,phi,pai,top=main()
#pos,neg,pos_words,neg_words=lookintofiles(file_path,stop_path)
def calculate_accuracy(predict_pai,pos_num,neg_num,rever=False):
    if rever:
        result_pred=predict_pai[:,0]<predict_pai[:,1]
    else:
        result_pred=predict_pai[:,0]>predict_pai[:,1]
    pos_accuracy=sum(result_pred[:pos_num])/pos_num
    neg_accuracy=1-sum(result_pred[pos_num:])/neg_num
    print("正类预测准确率为%f\n"%pos_accuracy)
    print("负类预测准确率为%f\n"%neg_accuracy)
    accuracy=(sum(result_pred[:pos_num])+neg_num-sum(result_pred[pos_num:]))/(pos_num+neg_num)
    return accuracy
def calculate_phi_distance(phi):
    '''
    计算与垃圾主题分布的KL距离
    '''
    litter=np.ones(phi.shape[:2])
    litter/=litter.shape[0]
    dist=[]
    for i in range(phi.shape[2]):
        distance=phi[:,:,i]*np.log(phi[:,:,i]/litter)
        distance=np.sum(distance)/phi.shape[1]
        dist.append(distance)  
    return dist
def parameter_tune(dpre,pos_num,neg_num,model_type="JST"):
    K_list=[10,20,30,40]
    time_consume=[]
    accuracy=[]
    dist=[]
    for k in K_list:
        if model_type=="ASUM":
            print("ASUM模型")
            asum=ASUM_model(dpre,K=k,alpha=50/k)
            start=time.clock()
            asum.Gibbs_sampling()
            end=time.clock()
            time_consume.append(end-start)
            print("本次训练所花时间%f"%(end-start))
            pai=asum.pai
            print("主题数为：%d"%k)
            accuracy.append(calculate_accuracy(pai,pos_num,neg_num))
            phi=asum.phi
            distance=calculate_phi_distance(phi)
            dist.append(distance)
        elif model_type=="JST":
            print("JST模型")
            jst=JST_model(dpre,K=k,alpha=50/k)
            start=time.clock()
            jst.Gibbs_sampling()
            end=time.clock()
            time_consume.append(end-start)
            print("本次训练所花时间%f"%(end-start))
            pai=jst.pai
            print("主题数为：%d"%k)
            accuracy.append(calculate_accuracy(pai,pos_num,neg_num))
            phi=jst.phi
            distance=calculate_phi_distance(phi)
            dist.append(distance)
        elif model_type=='JBST':
            print("JBST模型")
            jbst=JSBT_model(dpre,True,K=k,alpha=50/k)
            jbst.Gibbs_sampling()
            pai_docs=jbst.pai_docs
            accuracy.append(calculate_accuracy(pai_docs,pos_num,neg_num))
            phi=jbst.phi
            distance=calculate_phi_distance(phi)
            dist.append(distance)
        elif model_type=='STDP':
            print("STDP模型")
            stdp=STDP_model(dpre,True,K=k,alpha=50/k)
            stdp.Gibbs_sampling()
            pai=stdp.pai
            accuracy.append(calculate_accuracy(pai,pos_num,neg_num))
            phi=jbst.phi
            distance=calculate_phi_distance(phi)
            dist.append(distance)
        elif model_type=="MY_model":
            print("MY_model")
            my=My_model(dpre,K=k,alpha=50/k,T=5)
            my.Gibbs_sampling()
            pai=my.pai
            accuracy.append(calculate_accuracy(pai,pos_num,neg_num,True))
            phi=my.phi
            distance=calculate_phi_distance(phi)
            dist.append(distance)
        else:
            raise ValueError("模型不存在")
#    plt.subplot(211)
#    plot_time(K_list,time_consume)
#    plt.tight_layout(3)
#    plt.subplot(212)
    plot_accuracy(K_list,accuracy)
    return dist,accuracy
def plot_time(x,y):
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.ylabel(u"训练所花费时间")
    plt.xlabel(u"主题数目")
    plt.title(u"训练主题时间图")
    plt.plot(x,y)
def plot_accuracy(topic_num,accuracy):
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.ylabel("准确率")
    plt.xlabel(u"主题数目")
    plt.title(u"主题数目-准确率图")
    plt.plot(topic_num,accuracy)
def main_asum(tune=False,distance=False):
    corpus_sentence,pos_num,neg_num=load_sentence(file_path)
    corpus_sentence=combine(corpus_sentence)
    corpus_words=list(map(fenci_paragraph,corpus_sentence))
    corpus_words=filter_words(corpus_words,3)
    a=preprocess2(corpus_words,word2idmap_sentence,id2wordmap_sentence)
    if tune:
        print("开始对参数K调试")
        return parameter_tune(a,pos_num,neg_num,"ASUM")
    else:
        asum=ASUM_model(a,K=20,alpha=1,beta=0.01,gamma=0.1)
        asum.Gibbs_sampling()
        top=showtopN_words(a.id2word,asum.phi,15)
        theta,pai,phi=asum.theta,asum.pai,asum.phi
        accuracy=calculate_accuracy(pai,pos_num,neg_num)  #计算正负两类概率
        return theta,pai,phi,top,accuracy
        
#    return top,theta,pai
#theta,pai,phi,top,accuracy=main_asum()
#dist,accuracy=main_asum(True)
#with open("E:/硕士毕业论文/图片/asum模型准确率.pkl",'wb') as f:
#    pickle.dump(accuracy,f)
def write_topNwords(topN_words,filepath):
    with codecs.open(filepath,'w',"utf-8") as f:
        for i in range(len(topN_words)):
            f.write("主题%d：\n"%i)
            for l in range(len(topN_words[i])):
                f.write("情感%d"%l)
                for word in topN_words[i][l]:
                    f.write(word+"\t")
                f.write("\n")
                

    
        
                   
def sum_of(List):
    '''
    统计有多少个词语
    '''
    return sum(map(len,List))    
def main_jst(tune=False):
    pos,neg,pos_num,neg_num=load_data(file_path)
    corpus=pos+neg
    corpus_words=fenci(corpus,stop_path,3)
    a=preprocess(corpus_words,wordidmap_file,idwordmap_file)
    topN_file=r"E:/文本特征提取//topwords_jst.txt"
    
    if tune:
        return parameter_tune(a,pos_num,neg_num,"JST")
    else:
        jst=JST_model(a,K=20,alpha=50/20,beta=0.01)
        jst.Gibbs_sampling()
        top=showtopN_words(a.id2word,jst.phi,15)
        write_topNwords(top,topN_file)
        theta,pai,phi=jst.theta,jst.pai,jst.phi
        accuracy=calculate_accuracy(pai,pos_num,neg_num)  #计算正负两类概率
        return theta,pai,phi,top,accuracy
#theta,pai,phi,top,accuracy=main_jst()
#dist_jst,accuracy=main_jst(True)
#with open("E:/硕士毕业论文/图片/jst调参.pkl",'wb') as f:
#    pickle.dump(accuracy,f)
def fenci_withtags(docs,stop_path,freq=0):
    import jieba.posseg as pseg
    stop_words=[]
    words=[]
    tags=[]
    jieba.load_userdict("E:/文本特征提取/谭松波--酒店评论语料/user_dict.txt")
    with codecs.open(stop_path,'r','utf-8') as f:
        for line in f.readlines():    
            stop_words.append(line.strip())
    word_dict={}
    for review in docs:
        words_temp=[]
        tags_temp=[]
        for sentence in review:
            cut=pseg.cut(sentence.strip())
            for word,flag in cut:
                if word not in stop_words and word !=" ":
                    if word in word_dict:
                        word_dict[word]+=1
                    else:
                        word_dict[word]=1
                    words_temp.append(word)
                    tags_temp.append(flag)
        if len(words_temp)>0:
            words.append(words_temp)
            tags.append(tags_temp)
    count=Counter(word_dict).most_common()
    words_save={}.fromkeys([word for word,num in count if num>=freq])
    words_remain=[]
    tags_remain=[]
    for review_id in range(len(words)):
        words_temp=[]
        tags_temp=[]
        for word_num in range(len(words[review_id])):
            if words[review_id][word_num] in words_save:
                words_temp.append(words[review_id][word_num])
                tags_temp.append(tags[review_id][word_num])
        if len(words_temp)>0 and len(tags_temp)>0:
            words_remain.append(words_temp)
            tags_remain.append(tags_temp)
    tags_group=list(map(tag_groups,tags_remain))
    return words_remain,tags_remain,tags_group

def tag_groups(tags_list):
    tag_group=[]
    for tag in tags_list:
        if tag[0]=='a':
            tag_group.append(0)
        elif tag[0]=='d':
            tag_group.append(1)
        elif tag[0]=='n':
            tag_group.append(2)
        elif tag[0]=='v':
            tag_group.append(3)
        else:
            tag_group.append(4)
    return tag_group
class Article3(object):
    def __init__(self):
        self.words=[]
        self.tags=[]
        self.length=0
class Documents3(object):
    def __init__(self):
        self.docs_count =0
        self.words_count =0
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
def preprocess3(words_list,tags_list,word2id_file,id2word_file):
        docs=Documents3()
        words_idx=0
        for review_num in range(len(words_list)):
            temp=words_list[review_num]
            tags_temp=tags_list[review_num]
            article=Article3()
            for word_num in range(len(temp)):
                if temp[word_num] in docs.word2id:
                    article.words.append(docs.word2id[temp[word_num]])
                else:
                    docs.word2id[temp[word_num]] = words_idx
                    docs.id2word[words_idx]=temp[word_num]
                    article.words.append(words_idx)
                    words_idx += 1
                article.tags.append(tags_temp[word_num])
            article.length=len(temp)
            docs.docs.append(article)
        docs.docs_count = len(docs.docs)
        docs.words_count = len(docs.word2id)
        logger.info(u"共有%s个文档" % docs.docs_count)
        docs.cachewordidmap(word2id_file,id2word_file)
        logger.info(u"词与序号对应关系已保存到%s" % word2id_file)
        logger.info(u"序号与词对应关系已保存到%s" % id2word_file)
        return docs
class STDP_model(object):
    def __init__(self,dpre,seed=1,K=10,S=3,alpha=5,beta=0.01,gamma=[1,1],delta=[1,1],p=5,iterations=1000):
        '''
        先初始化超参数K（主题数），两个狄利克雷分布的参数alpha，beta
        '''
        self.dpre = dpre #获取预处理参数
        self.K=K
        self.alpha=alpha
        self.beta=set_parameter_beta(dpre.id2word,seed_path).T  #beta要是V*S
        self.gamma=np.array(gamma)
        self.V=dpre.words_count
        self.S=S
#        assert(len(self.gamma)==self.S)
        self.phi=np.zeros((self.V,self.K,self.S))
        self.theta=np.zeros((self.dpre.docs_count,self.K,self.S))
        self.pai=np.zeros((self.dpre.docs_count,2))
        self.iter=iterations
        self.nstw = np.zeros((self.V,self.K,self.S),dtype="int")#词i被分配到情感j,主题k的数量    
        self.nstwsum = np.zeros((self.K,self.S),dtype="int")    #被分配到情感j，主题k的词数量
        self.nstd = np.zeros((dpre.docs_count,self.K,self.S),dtype="int")  #第i篇文档中被分配到情感j，主题k的词数量     
        self.nsdsum = np.zeros((dpre.docs_count,self.S),dtype="int")    #第i篇文档情感j的数量
        self.ndsum = np.zeros((dpre.docs_count,2),dtype="int")  #每篇文档词数量
        self.delta=np.array(delta) #是p*2矩阵
        self.npw=np.zeros((p,2),dtype='int')
        self.npwsum=np.zeros((p,1),dtype='int')
        self.tags=np.array([ self.dpre.docs[x].tags for x in range(self.dpre.docs_count)])
        np.random.seed(seed)
        self.Z = np.array([ [[np.random.randint(0,self.K),np.random.randint(0,self.S)] \
                         for y in range(self.dpre.docs[x].length)] for x in range(self.dpre.docs_count)])
        for doc in range(len(self.Z)):
            for word in range(len(self.Z[doc])):
                topic_index=self.Z[doc][word][0]
                sentiment_index= self.Z[doc][word][1]
                group=self.tags[doc][word]
                if sentiment_index<2:
                    self.npw[group][0]+=1
                    self.ndsum[doc,0]+=1
                else:
                    self.npw[group][1]+=1
                    self.ndsum[doc,1]+=1
                self.npwsum[group]+=1
                self.nstw[self.dpre.docs[doc].words[word]][topic_index][sentiment_index]+=1
                self.nstwsum[topic_index][sentiment_index]+=1
                self.nstd[doc][topic_index][sentiment_index]+=1
                self.nsdsum[doc][sentiment_index]+=1
    def Gibbs_sampling(self):
    #,thetafile,phifile,parafile,topNfile,tassginfile
        np.random.seed(1)
        print("训练开始")
        for i in range(self.iter):
            if i %100==0:
                print("完成%d次迭代"%i)
            for m in range(self.dpre.docs_count):
                for word in range(len(self.Z[m])):
                    topic,sentiment=self.Z[m][word]
                    group=self.tags[m][word]
                    if sentiment<2:
                        self.npw[group][0]-=1
                        self.ndsum[m,0]-=1
                    else:
                        self.npw[group][1]-=1
                        self.ndsum[m,1]-=1
                    self.npwsum[group]-=1
                    word_id=self.dpre.docs[m].words[word]
                    self.nstw[word_id][topic][sentiment]-=1
                    self.nstd[m][topic][sentiment]-=1
                    self.nstwsum[topic][sentiment]-=1
                    self.nsdsum[m][sentiment]-=1
                    flag_true=(self.npw[group,0]+self.delta[group,0])/(self.npwsum[group]+sum(self.delta[group]))
                    flag_false=(self.npw[group,1]+self.delta[group,1])/(self.npwsum[group]+sum(self.delta[group]))
#                    flag_p=sum(flag_true+flag_false)
#                    u_flag=np.random.uniform(flag_sum)
                    flag=np.random.choice(2,1,p=[flag_false[0],flag_true[0]])
#                    if u_flag>flag_true:
#                        flag=0
#                    else:
#                        flag=1
                    if flag[0]:
                        self.p=(self.nstd[m,:,:2]+self.alpha)/(self.nsdsum[m,:2]+self.alpha*self.K)\
                        *(self.nstw[word_id,:,:2]+self.beta[word_id,:2])/(self.nstwsum[:,:2]+np.sum(self.beta[:,:2],axis=0))\
                        *(self.nsdsum[m,:2]+self.gamma)/(self.ndsum[m,0]+self.gamma*2)\
                        *(self.npw[group,0]+self.delta[group,0])/(self.npwsum[group]+sum(self.delta[group]))
                    else:
                        self.p=(self.nstd[m,:,2]+self.alpha)/(self.nsdsum[m,2]+self.alpha*self.K)\
                        *(self.nstw[word_id,:,2]+self.beta[word_id,2])/(self.nstwsum[:,2]+np.sum(self.beta[:,2],axis=0))\
                        *(self.npw[group,1]+self.delta[group,1])/(self.npwsum[group]+sum(self.delta[group]))
#                    prob=np.concatenate((self.p_true.flatten(),self.p_false.flatten()))
#                    assert(len(prob)==self.K*self.S)
                    prob=self.p.flatten()
                    for i in range(1,len(prob)):
                        prob[i]+=prob[i-1]
                    u=np.random.uniform(0.0,prob[-1])
                    for senti_top in range(len(prob)):
                        if prob[senti_top]>u:
                            break
                    if flag:
                        assert(len(prob)==self.K*2)               
                        new_sentiment=senti_top%2
                        new_topic=senti_top//2
                        self.npw[group][0]+=1
                        self.ndsum[m,0]+=1
                    else:
                        assert(len(prob)==self.K)
                        new_sentiment=2
                        new_topic=senti_top
                        self.npw[group][1]+=1
                        self.ndsum[m,1]+=1
#                    if new_sentiment<2:
#                        self.npw[group][0]+=1
#                        self.ndsum[m,0]+=1
#                    else:
#                        self.npw[group][1]+=1
#                        self.ndsum[m,1]+=1
                    self.nstw[word_id][new_topic][new_sentiment]+=1
                    self.nstwsum[new_topic][new_sentiment]+=1
                    self.nstd[m][new_topic][new_sentiment]+=1
                    self.nsdsum[m][new_sentiment]+=1
                    
                    self.npwsum[group]+=1
                    self.Z[m][word]=[new_topic,new_sentiment]
        print("训练结束")
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
            self.phi[i]=(self.nstw[i]+self.beta[i])/(self.nstwsum+np.sum(self.beta,axis=0))
    def _pai(self):
        for i in range(self.dpre.docs_count):
            self.pai[i]=(self.nsdsum[i,:2]+self.gamma)/(self.ndsum[i,0]+2*self.gamma)


def set1(word,pos_seed,neg_seed,polarity):
    if polarity==0:
        if word in pos_seed:
            return 0.1
        elif word in neg_seed:
            return 0.001
        else:
            return 0.01
    elif polarity==1:
        if word in pos_seed:
            return 0.001
        elif word in neg_seed:
            return 0.1
        else:
            return 0.01
    else:
        if word in neg_seed or word in pos_seed:
            return 0.001
        else:
            return 0.01

def set_parameter_beta(vocabulary,S,seed_path):
    '''
    设定STDP的beta先验参数
    '''
    beta=np.zeros((S,len(vocabulary)))
    neg_seed,pos_seed=seed_words(seed_path)
    for polarity in range(beta.shape[0]):
        beta[polarity]=np.array(list(map(set1,list(vocabulary.values()),[pos_seed]*len(vocabulary),[neg_seed]*len(vocabulary),[polarity]*len(vocabulary))))
    return beta
def set_parameter_delta(group_num,sigma,delta_strength=10):
    if not (min(sigma)>0 and max(sigma)<1):
        raise ValueError("sigma参数不对，（sigma取值在（0,1））")
    assert(len(sigma)==5)
    sigma1=np.array(sigma).reshape((5,1))
    sigma2=1-sigma1
    delta=np.concatenate((sigma1,sigma2),axis=1)
    return delta
def calculate_KL_dist(p,q):
    '''
    p,q的最好是行向量
    '''
    return np.sum(p*np.log(p/q))
def find_topic_stdp(phi):
    phi_S=phi[:,:,:2]
    ordinary=phi[:,:,2]
    topic_id=[]
    for i in range(phi_S.shape[2]):
        temp=[]
        for j in range(ordinary.shape[1]):
            dist=np.apply_along_axis(calculate_KL_dist,1,phi_S[:,:,i].T,ordinary.T[j])
            temp.append(dist.argmin())
        topic_id.append(temp)
    return topic_id
    
def main_sdtp(tune=False):
    pos,neg,pos_num,neg_num=load_data(file_path)
    corpus=pos+neg
    corpus_words,tags,tags_group=fenci_withtags(corpus,stop_path,3)
    a=preprocess3(corpus_words,tags_group,wordidmap_file,idwordmap_file)
    delta=set_parameter_delta(5,[0.9,0.9,0.7,0.9,0.1])
    beta=set_parameter_beta(a.id2word,seed_path)
    stdp=STDP_model(a,delta=delta,beta=beta)
    stdp.Gibbs_sampling()
    theta,pai,phi=stdp.theta,stdp.pai,stdp.phi
    accuracy=calculate_accuracy(pai,pos_num,neg_num)
    top=showtopN_words(a.id2word,stdp.phi,15)
    return theta,pai,phi,top,accuracy
#theta,pai,phi,top,accuracy=main_sdtp()            
       
def create_pair(words_list,width=5):
    
    if len(words_list)==1:
        return [(words_list[0],words_list[0])]
    elif len(words_list)<=width:
        return [(w1,w2) for id1,w1 in enumerate(words_list) for id2,w2 in enumerate(words_list) if id1<id2]
    else:
        begin=0
        end=width
        temp=[]
        while(begin<len(words_list)):
            temp.append(create_pair(words_list[begin:end]))
            begin+=width
            end+=width
        return sum(temp,[])
def turn_into_pair(dpre,width=5):
    words_pair=[]
    for i in range(dpre.docs_count):
        words_pair.append(dpre.docs[i].words)
    return list(map(create_pair,words_pair))
def put_label(phi,theta,pai):
    assert(phi.shape==theta.shape)
    prob=np.sum(theta*phi*pai,axis=0)/np.sum(phi*theta*pai)
    return prob
def put_label2(phi1,phi2,theta,pai):
    assert(phi1.shape==theta.shape and phi2.shape==theta.shape)
    prob=np.sum(phi1*phi2*theta*pai,axis=0)/np.sum(phi1*phi2*theta*pai)
    return prob
def output_label(sentiment_index):
    positive=sum(sentiment_index)/len(sentiment_index)
    negtive=1-positive
    return [positive,negtive]
    
class JSBT_model():
    def __init__(self,dpre,prio=False,seed=1,K=5,S=2,alpha=10,beta=0.01,gamma=1,iterations=1000):
        self.dpre = dpre #获取预处理参数
        self.biterm=turn_into_pair(dpre,10)
        self.K=K
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.V=dpre.words_count
        self.S=S
        self.phi=np.zeros((self.V,self.K,self.S))
        self.theta=np.zeros((self.K,self.S))
        self.pai=np.zeros((1,self.S))
        self.iter=iterations
        self.p=np.zeros((self.K,self.S))
        self.nstw = np.zeros((self.V,self.K,self.S),dtype="int")#词i被分配到情感j,主题k的数量
        self.nstwsum=np.zeros((self.K,self.S),dtype="int")
        self.nbsum = np.zeros((self.K,self.S),dtype="int")    #被分配到情感j，主题k的词对数量
#        self.n=sum(map(len,self.biterm))    
        self.n=0
        self.nsdsum = np.zeros((1,self.S),dtype="int")    #属于情感j的词对数量
        if prio:
            self.initialize_Z(seed_path)
        else:
            self.Z = np.array([ [[np.random.randint(0,self.K),np.random.randint(0,self.S)] \
                         for y in range(len(self.biterm[x]))] for x in range(len(self.biterm))])
        self.origin_Z=np.array([ self.dpre.docs[x].words for x in range(self.dpre.docs_count)])
        self.Z_count=[[] for x in range(self.dpre.docs_count)]
        self.pai_docs=np.array((dpre.docs_count,self.S),dtype='float')
        self.origin_Z_count=[[] for x in range(len(self.origin_Z))]
        for doc in range(len(self.Z)):
            biterm_count={}
            for word in range(len(self.Z[doc])):
                topic_index,sentiment_index=self.Z[doc][word]
                word1,word2=self.biterm[doc][word]
                if (word1,word2) in biterm_count or (word2,word1) in biterm_count:
                    if biterm_count.get((word1,word2)):
                        biterm_count[(word1,word2)]+=1
                    else:
                        biterm_count[(word2,word1)]+=1
                else:
                    biterm_count[(word1,word2)]=1
#                sentiment_index= self.Z[doc][word][1]
                self.nstw[word1][topic_index][sentiment_index]+=1
                self.nstw[word2][topic_index][sentiment_index]+=1
                self.nstwsum[topic_index][sentiment_index]+=2
                self.nbsum[topic_index][sentiment_index]+=1
                self.nsdsum[0,sentiment_index]+=1
                self.n+=1
            self.Z_count[doc].append(biterm_count)
        for doc in range(len(self.origin_Z)):
            word_count={}
            for word in range(len(self.origin_Z[doc])):
                if self.origin_Z[doc][word] in word_count:
                    word_count[self.origin_Z[doc][word]]+=1
                else:
                    word_count[self.origin_Z[doc][word]]=1
            self.origin_Z_count[doc].append(word_count)
    def Gibbs_sampling(self):
        print("训练开始")
        for i in range(self.iter):
            if i%100==0 and i>0:
                print("完成%d次迭代"%i)
            for m in range(len(self.Z)):
                for word in range(len(self.Z[m])):
                    topic,sentiment=self.Z[m][word]
                    word_id1,word_id2=self.biterm[m][word]
                    self.nstw[word_id1][topic][sentiment]-=1
                    self.nstw[word_id2][topic][sentiment]-=1
                    self.nbsum[topic][sentiment]-=1
                    self.nstwsum[topic][sentiment]-=2
                    self.nsdsum[0,sentiment]-=1
                    self.n-=1
                    self.p=(self.nbsum+self.alpha)/(self.nsdsum+self.K*self.alpha)\
                       *(self.nstw[word_id1]+self.beta)*(self.nstw[word_id2]+self.beta)\
                       /((self.nstwsum+self.V*self.beta)*(self.nstwsum+self.V*self.beta+1))\
                       *(self.nsdsum+self.gamma)/(self.n+self.gamma*self.S)
                    
#                    self.p=((self.nbsum+self.alpha)*(self.nstw[word_id1]+self.beta.T[word_id1])\
#                    *(self.nstw[word_id2]+self.beta.T[word_id2])*(self.nsdsum+self.gamma))\
#                    /((self.nstwsum+np.sum(self.beta.T,axis=0))*(self.nstwsum+np.sum(self.beta.T,axis=0)+1)\
#                      *(self.nsdsum+self.K*self.alpha)*(self.n+self.gamma*self.S))
                    prob=self.p.flatten()
                    assert(len(prob)==self.K*self.S)
                    for i in range(1,len(prob)):
                        prob[i]+=prob[i-1]
                    np.random.seed(1)
                    u=np.random.uniform(0.0,prob[-1])
                    for senti_top in range(len(prob)):
                        if prob[senti_top]>u:
                            break 
#                    prob/=np.sum(prob)
#                    senti_top=np.random.choice(self.K*self.S,1,p=prob)[0]                
                    new_sentiment=senti_top%self.S
                    new_topic=senti_top//self.S
                    self.nstw[word_id1][new_topic][new_sentiment]+=1
                    self.nstw[word_id2][new_topic][new_sentiment]+=1
                    self.nstwsum[new_topic][new_sentiment]+=2
                    self.nbsum[new_topic][new_sentiment]+=1
#                    self.nstd[m][new_topic][new_sentiment]+=1
                    self.nsdsum[0,new_sentiment]+=1
                    self.n+=1
#                    self.ndsum[m]+=1
                    self.Z[m][word]=[new_topic,new_sentiment]
        logger.info(u"迭代完成。")
        print("训练结束")
        
        logger.debug(u"计算文章-主题分布")
        self._theta()
        logger.debug(u"计算词-主题分布")
        self._phi()
        logger.debug("计算文档-情感分布")
        self._pai()
    def initialize_Z(self,seed_path):
        neg_seed,pos_seed=seed_words(seed_path)
        self.Z = np.array([ [[np.random.randint(0,self.K),self.initial_S(self.biterm[x][y],neg_seed,pos_seed)] \
                         for y in range(len(self.biterm[x]))] for x in range(len(self.biterm))])
    def initial_S(self,word_pair,neg,pos):
        w1,w2=word_pair
        if w1 in pos and w2 not in neg:
            score=2
        elif w1 in neg and w2 not in pos:
            score=-2
        else:
            score=0
        if score>0:
            return 0
        elif score<0:
            return 1
        else:
            return np.random.randint(0,self.S)
        
    def _theta(self):
#        for i in range(self.dpre.docs_count):
        self.theta=(self.nbsum+self.alpha)/(self.nsdsum+self.K*self.alpha)
    def _phi(self):
        for i in range(self.V):
            self.phi[i]=(self.nstw[i]+self.beta)/(self.nstwsum+self.V*self.beta)
    def _pai(self):
#        for i in range(self.dpre.docs_count):
        self.pai=(self.nsdsum+self.gamma)/(self.n+self.S*self.gamma)
#        label=[]
#        for i in range(len(self.phi)):
#            label.append(put_label(self.phi[i],self.theta,self.pai))
#        label=np.array(label)
#        sentiment=[]
#        for i in range(len(self.origin_Z)):
#            sentiment.append(output_label(list(label[self.origin_Z[i]])))
#        self.pai_docs=np.array(sentiment)
        sentiment=[]
        for i in range(len(self.Z_count)):
            p=0
            w_total=sum(self.Z_count[i][0].values())
            for w1,w2 in self.Z_count[i][0].keys():
                prob1=put_label2(self.phi[w1],self.phi[w2],self.theta,self.pai)
                p+=prob1*self.Z_count[i][0][(w1,w2)]/w_total
            sentiment.append(p)
        self.pai_docs=np.array(sentiment)
#        sentiment=[]
#        for i in range(len(self.origin_Z)):
#            p=0
#            w_total=sum(self.origin_Z_count[i][0].values())
#            for w in self.origin_Z_count[i][0].keys():
#                prob=put_label(self.phi[w],self.theta,self.pai)
#                p+=prob*self.origin_Z_count[i][0][w]/w_total
#            sentiment.append(p)
#        self.pai_docs=np.array(sentiment)
        
        
        
def main_jbst(tune=False):
    pos,neg,pos_num,neg_num=load_data(file_path,True,10,10)
    corpus=pos+neg
    corpus_words=fenci(corpus,stop_path,3)
    a=preprocess(corpus_words,wordidmap_file,idwordmap_file)
#    beta=set_parameter_beta(a.id2word,2,seed_path)
#    topN_file=r"E:/文本特征提取//topwords_jst.txt"
    
    if tune:
        parameter_tune(a,pos_num,neg_num,"JBST")
    else:
        jsbt=JSBT_model(a,S=2,K=5,alpha=50/5,beta=0.01)
        jsbt.Gibbs_sampling()
#        top=showtopN_words(a.id2word,jsbt.phi,15)
#        write_topNwords(top,topN_file)
        theta,pai,phi,pai_docs=jsbt.theta,jsbt.pai,jsbt.phi,jsbt.pai_docs
        accuracy=calculate_accuracy(pai_docs,pos_num,neg_num)  #计算正负两类概率
        top=showtopN_words(a.id2word,phi,15)
        return theta,pai,phi,pai_docs,accuracy,top   
#theta1,pai1,phi1,pai_docs1,accuracy,top=main_jbst()
#dist=main_jbst(True)
#label=[]
#sentiment=[]
#temp=[]
#for i in range(len(phi1)):
#    label.append(put_label(phi1[i],theta1,pai1))
#label=np.array(label)
#origin_Z=np.array([ a.docs[x].words for x in range(a.docs_count)])        
#for i in range(len(origin_Z)):
#    sentiment.append(output_label(list(label[origin_Z[i]])))
#for i in jieba.cut("不干净"):
#    print("/".join(i))    
#for i in temp[:6]:
#    print(output_label(i))
#pos,neg,pos_num,neg_num=load_data(file_path,True,70,70)
#corpus=pos+neg
#corpus_words=fenci(corpus,stop_path,3)
#a=preprocess(corpus_words,wordidmap_file,idwordmap_file)
#jsbt=JSBT_model(a,True,S=2,K=5,alpha=50/5,beta=0.01)
#jsbt.Gibbs_sampling()
#theta,pai,phi,pai_docs=jsbt.theta,jsbt.pai,jsbt.phi,jsbt.pai_docs
#accuracy=calculate_accuracy(pai_docs,pos_num,neg_num)
#top=showtopN_words(a.id2word,phi,15)
def create_window(doc,win=2):
    sentence_num=len(doc)
    win_total=sentence_num+win-1
    v=[]
    for i in range(sentence_num):
        v_temp=[]
        for j in range(win):
            v_temp.append(i+j)
        v.append(v_temp)
    return v,win_total
def count_sentence_word(doc):
    return list(map(len,doc))
def initial_K(T,S,K_loc,K_gl):
    r=np.random.randint(2)
    v=np.random.randint(T)
    s=np.random.randint(S)
    if r:
        return [v,r,np.random.randint(K_loc),s]
    else:
        return [v,r,np.random.randint(K_gl),s]
def main_multi_grain():
    corpus_sentence,pos_num,neg_num=load_sentence(file_path,True,20,20)
    corpus_sentence=combine(corpus_sentence)
    corpus_words=list(map(fenci_paragraph,corpus_sentence))
    corpus_words=filter_words(corpus_words,3)
    windows=np.array(list(map(create_window,corpus_words)))
    windows_total=list(windows[:,-1])
    windows=windows[:,:-1].tolist()
    windows=sum(windows,[])
    a=preprocess2(corpus_words,word2idmap_sentence,id2wordmap_sentence)
    multi_model=Multi_model(a,windows_total,K_loc=2,K_gl=4)
    multi_model.Gibbs_sampling()
    pai,phi_gl,phi_loc=multi_model.pai,multi_model.phi_gl,multi_model.phi_loc
    top_gl=showtopN_words(a.id2word,phi_gl,15)
    top_loc=showtopN_words(a.id2word,phi_loc,15)
    accuracy=calculate_accuracy(pai,pos_num,neg_num)
    return pai,phi_gl,phi_loc,top_gl,top_loc,accuracy
class Multi_model():
    def __init__(self,dpre,windows_num,initialize=True,K_gl=10,K_loc=5,S=2,T=2,seed=1,iterations=1000,alpha_gl=50/10,alpha_loc=50/5,beta=0.01,gamma=1,alpha_mix=0.1,ita=0.1):
        self.dpre=dpre
        self.ita=ita
        self.K_gl=K_gl
        self.K_loc=K_loc
        self.S=S
        self.T=T
        self.iter=iterations
        self.alpha_gl=alpha_gl
        self.alpha_loc=alpha_loc
        self.alpha_mix=alpha_mix
        self.beta=beta
        self.gamma=gamma
        self.V=dpre.words_count
        self.phi_loc=np.zeros((self.V,self.K_loc,self.S))
        self.phi_gl=np.zeros((self.V,self.K_gl,self.S))
        self.theta_loc=np.zeros((dpre.docs_count,self.K_loc,self.S))
        self.theta_gl=np.zeros((dpre.docs_count,self.K_gl,self.S))
        self.pai=np.zeros((dpre.docs_count,self.S))
        self.p_loc=np.zeros((self.K_loc,self.S))
        self.p_gl=np.zeros((self.K_gl,self.S))
        self.nstw_gl = np.zeros((self.V,self.K_gl,self.S),dtype="float")#词i被分配到情感j,主题k的数量    
        self.nstw_loc = np.zeros((self.V,self.K_loc,self.S),dtype="float")
        self.nstwsum_gl = np.zeros((self.K_gl,self.S),dtype="float")    #被分配到情感j，主题k的词数量
        self.nstwsum_loc = np.zeros((self.K_loc,self.S),dtype="float") 
        self.nstd_gl = np.zeros((dpre.docs_count,self.K_gl,self.S),dtype="float")  #第i篇文档中被分配到情感j，全局主题k的词数量     
        self.nsd_gl=np.zeros((dpre.docs_count,self.S))  #文档d中分配到情感l和全局主题的词数
        self.nstvd_loc = [np.zeros((x,self.K_loc,self.S)) for x in windows_num ]  #第d篇文档中被分配到滑动窗口v且情感j，局部主题为k的词数量
        self.nsvd_loc=[np.zeros((x,self.S))   for x in windows_num] #文档d中情感为l且为局部主题在窗口为v的词数
        self.nsdsum = np.zeros((dpre.docs_count,self.S),dtype="float")    #每篇文档的不同情感词汇数量
        self.ndvsum_sentence=[np.zeros((dpre.docs[x].sentence,T),dtype='int')  for x in range(dpre.docs_count)] #文档d中句子s被分到滑动窗口v的词数
        self.ndwsum = np.zeros(dpre.docs_count,dtype="float")  #每篇文档词语数量
        self.ndvsum=[np.zeros((x,2),dtype='int') for x in windows_num] #每个窗口内gl和loc的数量
        self.ndvwsum=[np.zeros((x,1),dtype='int') for x in windows_num ] #每个窗口内的总词数
        self.nstvsum_loc=[[np.zeros((self.K_loc,self.S),dtype='int')]*windows_num[x] for x in range(dpre.docs_count)]#文档d的滑动窗口v被分配到局部主题z,情感l的词数
#        self.nstvsum_gl=[[np.zeros((self.K_gl,self.S),dtype='int')]*windows_num[x] for x in range(dpre.docs_count)]
        self.nstwsum_sentence=[np.zeros(dpre.docs[x].sentence) for x in range(dpre.docs_count)] #每个句子的词汇数
        assert(len(self.ndvsum)==len(self.ndvwsum))
#        np.random.seed(seed)
#        if initialize:
#            self.initialize_Z(seed_path)
#        else:
#        self.Z = np.array([ [[np.random.randint(0,self.T+y),np.random.randint(0,2)] \
#                         for y in range(self.dpre.docs[x].sentence)] for x in range(self.dpre.docs_count)])  
        self.Z=np.array([ [[initial_K(self.T,2,self.K_loc,self.K_gl) for z in range(len(self.dpre.docs[x].words[y])) ]\
                         for y in range(self.dpre.docs[x].sentence)] for x in range(self.dpre.docs_count)]) 
#        self.Z_word= np.array([ [self.dpre.docs[x].words[y] \
#                         for y in range(self.dpre.docs[x].sentence)] for x in range(self.dpre.docs_count)])
        self.Z_word=np.array([dpre.docs[x].words for x in range(dpre.docs_count)])
        for doc in range(len(self.Z)):
#            print("文档%d"%doc)
            for sentence in range(len(self.Z[doc])):
#                print("句子%d"%sentence)
                for word in range(len(self.Z[doc][sentence])):
#                    print("词%d"%word)
                    v_index,r_index,topic_index,sentiment_index=self.Z[doc][sentence][word]
#                    if v_index>=self.T:
#                        raise ValueError("v的赋值出错")
                    v_index+=sentence
#                    print('v取值%d'%v_index)
                    word_id=self.Z_word[doc][sentence][word]
                    if r_index:
                        self.nstw_loc[word_id][topic_index][sentiment_index]+=1
                        self.nstwsum_loc[topic_index][sentiment_index]+=1
                        self.ndvsum[doc][v_index][1]+=1
                        self.nstvd_loc[doc][v_index][topic_index][sentiment_index]+=1
                        self.nsvd_loc[doc][v_index][sentiment_index]+=1
                        
                    else:
                        self.nstwsum_gl[topic_index][sentiment_index]+=1
                        self.nstw_gl[word_id][topic_index][sentiment_index]+=1
#                        if v_index>=self.ndvsum[doc].shape[0]:
#                            print(doc,sentence,word)
                        self.ndvsum[doc][v_index][0]+=1
                        self.nstd_gl[doc][topic_index][sentiment_index]+=1
                        self.nsd_gl[doc][sentiment_index]+=1
#                    self.ndvsum[doc][v_index]+=1
                    self.ndvsum_sentence[doc][sentence][v_index-sentence]+=1
                    self.nsdsum[doc][sentiment_index]+=1
                    self.ndwsum[doc]+=1
                    self.ndvwsum[doc][v_index]+=1
                    self.nstwsum_sentence[doc][sentence]+=1
                    
                    
    def Gibbs_sampling(self):
#        np.random.seed(1)
        print("训练开始")
        for i in range(self.iter):
            if i %100==0 and i>0:
                print("已完成%d次迭代"%i)
            for doc in range(self.dpre.docs_count):
                for sentence in range(len(self.Z[doc])):
                    for word in range(len(self.Z[doc][sentence])):
                        v_index,r_index,topic_index,sentiment_index=self.Z[doc][sentence][word]
                        v_index+=sentence
                        word_id=self.Z_word[doc][sentence][word]
                        if r_index:
                            self.nstw_loc[word_id][topic_index][sentiment_index]-=1
                            self.nstwsum_loc[topic_index][sentiment_index]-=1
                            self.ndvsum[doc][v_index][1]-=1
                            self.nstvd_loc[doc][v_index][topic_index][sentiment_index]-=1
                            self.nsvd_loc[doc][v_index][sentiment_index]-=1
                        else:
                            self.nstwsum_gl[topic_index][sentiment_index]-=1
                            self.nstw_gl[word_id][topic_index][sentiment_index]-=1
                            self.ndvsum[doc][v_index][0]-=1
                            self.nstd_gl[doc][topic_index][sentiment_index]-=1
                            self.nsd_gl[doc][sentiment_index]-=1
                        self.ndvsum_sentence[doc][sentence][v_index-sentence]-=1
                        self.nsdsum[doc][sentiment_index]-=1
                        self.ndwsum[doc]-=1
                        self.ndvwsum[doc][v_index]-=1
                        self.nstwsum_sentence[doc][sentence]-=1
                        
                        prob_v=(self.ndvsum_sentence[doc][sentence]+self.ita)/(self.nstwsum_sentence[doc][sentence]+self.ita*self.T)
#                        if (prob_v<0).any():
#                            print(doc,sentence,word)
                        v_new=np.random.choice(self.T,1,p=prob_v)[0]
                        prob_r=(self.ndvsum[doc][v_new+sentence]+self.alpha_mix)/(2*self.alpha_mix+self.ndvwsum[doc][v_new+sentence])
                        r_new=np.random.choice(2,1,p=prob_r)[0]
#                        temp=(self.ndvsum_sentence[doc][sentence]+self.ita)/(self.nstwsum_sentence[doc][sentence]+self.ita*self.T)\
#                        *((self.ndvsum[doc][sentence:sentence+self.T]+self.alpha_mix)/(2*self.alpha_mix+self.ndvwsum[doc][sentence:sentence+self.T])).T
#                        p_temp=temp.flatten()
#                        
#                        for i in range(1,len(p_temp)):
#                            p_temp[i]+=p_temp[i-1]
#                        rv=np.random.uniform(0.0,p_temp[-1])
#                        for r_v in range(len(p_temp)):
#                            if p_temp[r_v]>rv:
#                                break
#                        r_new=r_v//self.T
#                        v_new=r_v%self.T
                        if r_new:
                            self.p_loc=(self.nstw_loc[word_id]+self.beta)/(self.nstwsum_loc+self.V*self.beta)\
                            *(self.nstvd_loc[doc][v_new+sentence]+self.alpha_loc)/(self.nsvd_loc[doc][v_new+sentence]+self.alpha_loc*self.K_loc)\
                            *(self.ndvsum[doc][v_new+sentence][1]+self.alpha_mix)/(self.ndvwsum[doc][v_new+sentence]+self.alpha_mix*2)\
                            *(self.nsdsum[doc]+self.gamma)/(self.ndwsum[doc]+self.S*self.gamma)\
                            *(self.ndvsum_sentence[doc][sentence][v_new]+self.ita)/(self.nstwsum_sentence[doc][sentence]+self.ita*self.T)
                            
                            prob=self.p_loc.flatten()
                            assert(len(prob)==self.K_loc*self.S)
                            assert((prob>0).all() and (prob!=np.inf).any())
                            for i in range(1,len(prob)):
                                prob[i]+=prob[i-1]
                            u=np.random.uniform(0.0,prob[-1])
                            for senti_top in range(len(prob)):
                                if prob[senti_top]>u:
                                    break
                            new_sentiment=senti_top%self.S
                            new_topic=senti_top//self.S
                            self.nstw_loc[word_id][new_topic][new_sentiment]+=1
                            self.nstwsum_loc[new_topic][new_sentiment]+=1
                            self.ndvsum[doc][v_new+sentence][1]+=1
                            self.nstvd_loc[doc][v_new+sentence][new_topic][new_sentiment]+=1
                            self.nsvd_loc[doc][v_new+sentence][new_sentiment]+=1
                            
                        else:
                            self.p_gl=(self.nstw_gl[word_id]+self.beta)/(self.nstwsum_gl+self.V*self.alpha_gl)\
                            *(self.nstd_gl[doc]+self.alpha_gl)/(self.nsd_gl[doc]+self.alpha_gl*self.K_gl)\
                            *(self.ndvsum[doc][v_new+sentence][0]+self.alpha_mix)/(self.ndvwsum[doc][v_new+sentence]+self.alpha_mix*2)\
                            *(self.nsdsum[doc]+self.gamma)/(self.ndwsum[doc]+self.gamma*self.S)\
                            *(self.ndvsum_sentence[doc][sentence][v_new]+self.ita)/(self.nstwsum_sentence[doc][sentence]+self.ita*self.T)
                            prob=self.p_gl.flatten()
                            assert(len(prob)==self.K_gl*self.S)
                            assert((prob>0).all() and (prob!=np.inf).any())
                            for i in range(1,len(prob)):
                                prob[i]+=prob[i-1]
                            u=np.random.uniform(0.0,prob[-1])
                            for senti_top in range(len(prob)):
                                if prob[senti_top]>u:
                                    break
                            new_sentiment=senti_top%self.S
                            new_topic=senti_top//self.S
                            self.nstw_gl[word_id][new_topic][new_sentiment]+=1
                            self.nstwsum_gl[new_topic][new_sentiment]+=1
                            self.ndvsum[doc][v_new+sentence][0]+=1
                            self.nstd_gl[doc][new_topic][new_sentiment]+=1
                            self.nsd_gl[doc][new_sentiment]+=1
                        self.ndvsum_sentence[doc][sentence][v_new]+=1
                        self.nsdsum[doc][new_sentiment]+=1
                        self.ndwsum[doc]+=1
                        self.ndvwsum[doc][v_new+sentence]+=1
                        self.nstwsum_sentence[doc][sentence]+=1
                        self.Z[doc][sentence][word]=[v_new,r_new,new_topic,new_sentiment]

        logger.info(u"迭代完成。")
        logger.debug(u"计算文章-主题分布")
#        self._theta()
#        logger.debug(u"计算词-主题分布")
        self._phi()
        logger.debug("计算文档-情感分布")
        self._pai()
#        logger.debug(u"保存模型")
#        self.save(thetafile,phifile,parafile,topNfile,tassginfile)
    def _theta(self):
        for i in range(self.dpre.docs_count):
            self.theta_loc[i]=(self.nstd[i]+self.alpha)/(self.nsdsum[i]+self.K*self.alpha)
    def _phi(self):
        for i in range(self.V):
            self.phi_loc[i]=(self.nstw_loc[i]+self.beta)/(self.nstwsum_loc+self.V*self.beta)
            self.phi_gl[i]=(self.nstw_gl[i]+self.beta)/(self.nstwsum_gl+self.beta*self.V)
    def _pai(self):
        for i in range(self.dpre.docs_count):
            self.pai[i]=(self.nsdsum[i]+self.gamma)/(self.ndwsum[i]+self.S*self.gamma)    
#pai,phi_gl,phi_loc,top_gl,top_loc,accuracy=main_multi_grain()


class My_model():
    def __init__(self,dpre,prior=False,seed=1,K=10,S=2,alpha=1,beta=0.01,gamma=1,T=3,ita=0.2,iterations=1000):
        self.dpre = dpre #获取预处理参数
        self.K=K
        self.T=T
        self.ita=ita
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.V=dpre.words_count
        self.S=S
        self.phi=np.zeros((self.V,self.K,self.S))
        self.theta=np.zeros((self.dpre.docs_count,self.K,self.S))
        self.pai=np.zeros((self.dpre.docs_count,self.S))
        self.iter=iterations
        self.p=np.zeros((self.K,self.S))
        self.ndsvt=np.zeros((dpre.docs_count,T,K,S))
        self.ndvsum=np.zeros((dpre.docs_count,T,S))
        self.nstw = np.zeros((self.V,self.K,self.S),dtype="int")#词i被分配到情感j,主题k的数量    
        self.nstwsum = np.zeros((self.K,self.S),dtype="int")    #被分配到情感j，主题k的词数量
        self.nvd = np.zeros((dpre.docs_count,T),dtype="int")  #第i篇文档中被分配到情感j，主题k的词数量     
        self.nsdsum = np.zeros((dpre.docs_count,self.S),dtype="int")    #第i篇文档情感j的数量
        self.ndsum = np.zeros(dpre.docs_count,dtype="int")  #每篇文档词数量
        np.random.seed(seed)
#        self.initialize_Z(seed_path)
        # M*doc.size()，文档中词的主题分布
        if prior:
            self.initialize_Z(seed_path)
        self.Z=np.array([[[np.random.randint(T),np.random.randint(K),np.random.randint(S)] for y in range(len(dpre.docs[x].words))] for x in range(dpre.docs_count)])
        for doc in range(len(self.Z)):
            for word in range(len(self.Z[doc])):
                v_index,topic_index,sentiment_index=self.Z[doc][word]
                word_id=self.dpre.docs[doc].words[word]
                self.nstw[word_id][topic_index][sentiment_index]+=1
                self.nstwsum[topic_index][sentiment_index]+=1
                self.ndsvt[doc][v_index][topic_index][sentiment_index]+=1
                self.ndvsum[doc][v_index][sentiment_index]+=1
                self.nvd[doc][v_index]+=1
                self.nsdsum[doc][sentiment_index]+=1
                self.ndsum[doc]+=1  
    def initialize_Z(self,seed_path):
        neg_seed,pos_seed=seed_words(seed_path)
        self.Z = np.array([ [[np.random.randint(self.T),np.random.randint(0,self.K),self.initial_S(self.dpre.id2word[self.dpre.docs[x].words[y]],neg_seed,pos_seed)] \
                         for y in range(self.dpre.docs[x].length)] for x in range(self.dpre.docs_count)])
    def initial_S(self,word,neg,pos):
        if word in neg:
            return 1
        elif word in pos:
            return 0
        else:
            return np.random.randint(0,self.S)
    def Gibbs_sampling(self):
    #,thetafile,phifile,parafile,topNfile,tassginfile
        np.random.seed(1)
        print("训练开始")
        for i in range(self.iter):
            if i%100==0 and i>0:
                print("完成%d次迭代"%i)
            for m in range(self.dpre.docs_count):
                for word in range(len(self.Z[m])):
#                    if i==1:
#                        return self.nvd,self.Z
                    v,topic,sentiment=self.Z[m][word]
                    word_id=self.dpre.docs[m].words[word]
                    self.nstw[word_id][topic][sentiment]-=1
                    self.nstwsum[topic][sentiment]-=1
                    self.ndsvt[m][v][topic][sentiment]-=1
                    self.ndvsum[m][v][sentiment]-=1
#                    print(self.nvd)
                    self.nvd[m][v]-=1
#                    print(self.nvd)
                    self.nsdsum[m][sentiment]-=1
                    self.ndsum[m]-=1
                    assert(np.sum(self.nvd)==np.sum(self.ndsum))
#                    if i==1:
#                        return self.nvd,self.ndsum
                    p_v=(self.nvd[m]+self.ita)/(self.ndsum[m]+self.ita*self.T)
#                    return p_v
                    if (p_v<0).any():
                        print(i,m,word)
                        print(p_v)
                    v_new=np.random.choice(self.T,1,p=p_v)[0]
#                    return v_new
                    self.p=(self.ndsvt[m][v_new]+self.alpha)/(self.ndvsum[m][v_new]+self.K*self.alpha)\
                       *(self.nstw[word_id]+self.beta)/(self.nstwsum+self.V*self.beta)\
                       *(self.nsdsum[m]+self.gamma)/(self.ndsum[m]+self.S*self.gamma)\
                       *(self.nvd[m][v_new]+self.ita)/(self.ndsum[m]+self.T*self.ita)
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
                    new_sentiment=senti_top%self.S
                    new_topic=senti_top//self.S
                    self.nstw[word_id][new_topic][new_sentiment]+=1
                    self.nstwsum[new_topic][new_sentiment]+=1
                    self.ndsvt[m][v_new][new_topic][new_sentiment]+=1
                    self.ndvsum[m][v_new][new_sentiment]+=1
                    self.nvd[m][v_new]+=1
                    self.nsdsum[m][new_sentiment]+=1
                    self.ndsum[m]+=1
                    self.Z[m][word]=[v_new,new_topic,new_sentiment]
        logger.info(u"迭代完成。")
        print("训练结束")
        
        logger.debug(u"计算文章-主题分布")
#        self._theta()
#        logger.debug(u"计算词-主题分布")
        self._phi()
        logger.debug("计算文档-情感分布")
        self._pai()
#        logger.debug(u"保存模型")
#        self.save(thetafile,phifile,parafile,topNfile,tassginfile)
#    def _theta(self):
#        for i in range(self.dpre.docs_count):
#            self.theta[i]=(self.nstd[i]+self.alpha)/(self.nsdsum[i]+self.K*self.alpha)
    def _phi(self):
        for i in range(self.V):
            self.phi[i]=(self.nstw[i]+self.beta)/(self.nstwsum+self.V*self.beta)
    def _pai(self):
        for i in range(self.dpre.docs_count):
            self.pai[i]=(self.nsdsum[i]+self.gamma)/(self.ndsum[i]+self.S*self.gamma)

def main_my(tune=False):
    pos,neg,pos_num,neg_num=load_data(file_path)
    corpus=pos+neg
    corpus_words=fenci(corpus,stop_path,3)
    a=preprocess(corpus_words,wordidmap_file,idwordmap_file)
#    topN_file=r"E:/文本特征提取//topwords_jst.txt"
    
    if tune:
        dist,accuracy=parameter_tune(a,pos_num,neg_num,"MY_model")
        return dist,accuracy
    else:
        my=My_model(a,True,K=30,alpha=50/30,T=5,beta=0.01)
        my.Gibbs_sampling()
        top=showtopN_words(a.id2word,my.phi,15)
#        write_topNwords(top,topN_file)
        pai,phi=my.pai,my.phi
        accuracy=calculate_accuracy(pai,pos_num,neg_num,True)  #计算正负两类概率
        return pai,phi,top,accuracy
pai,phi,top,accuracy=main_my()
#dist,accuracy=main_my(True)