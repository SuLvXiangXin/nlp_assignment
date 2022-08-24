#!/usr/bin/env python
# coding: utf-8

# # Assignment-01
# 顾淳 19307110344

# ## 0. 准备工作

# 导入相关库

# In[1]:

import time
time1 = time.time()
import json
import nltk
import numpy as np
import pandas  as pd
import pickle
import time
import os
import re
from math import log
from collections import defaultdict
from nltk import sent_tokenize, word_tokenize, ngrams, FreqDist, ConditionalFreqDist
import warnings
from IPython.display import display

warnings.filterwarnings("ignore")
import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(stream=sys.stdout)
# 导入数据

# In[2]:


np.random.seed(0)
wiki_data = []
with open("enwiki_20220201.json", "r") as f:
    for each_line in f:
        record = json.loads(each_line)
        wiki_data.append(record)
print(len(wiki_data))

# In[3]:


# wiki_data=wiki_data[:10]+wiki_data[-10:]


# ## Task-1

# ### 将数据导入 DataFrame

# In[4]:


df = pd.DataFrame(wiki_data).iloc[:10]
# 删除 wiki_data 释放内存
del wiki_data
df

# ### 对所有 text 进行分句、分词

# In[5]:


# 使用 sent_tokenize 函数对 text 进行分句
df['sentences'] = df['text'].apply(lambda t: sent_tokenize(t))
# 计算各 text 的句子数
df['num_sentences'] = df['sentences'].apply(lambda t: len(t))

# 使用 word_tokenize 函数对 sentence 进行分词
df['words'] = df['sentences'].apply(lambda t: [word_tokenize(sent) for sent in t if sent])
# 计算各 text 的单词数
df['num_words'] = df['words'].apply(lambda t: sum([len(sent) for sent in t]))
df

# ### 展示各 label 的数量、平均句数、平均单词数

# In[6]:


# 计算各 label 对应的文件数量
label_count = df['label'].groupby(df['label']).count()
# 计算平均句子数
sentences_average = df['num_sentences'].groupby(df['label']).mean()
# 计算平均单词数
words_average = df['num_words'].groupby(df['label']).mean()
# 将上述结果拼接方便查看
labels = pd.concat(
    {'count': label_count, 'sentences_average': sentences_average.round(2), 'words_average': words_average.round(2)},
    axis=1)
labels

# ### 定义预处理函数，对句子进行去特殊字符以及小写化处理

# In[7]:


import re


def preprocess(sentence):
    # 将句内除字母、数字、空格外的所有字符替换为空格
    res = re.sub(r'[^\w\s]', ' ', sentence)
    return res.lower()


# 用一个示例查看 preprocessing 函数的作用
print('Example:\n')
print('raw string:\n', df['sentences'][0][0], '\n')
print('preprocessing:\n', preprocess(df['sentences'][0][0]))


# ### 应用预处理函数更新 dataset 中的 sentences 和 words

# In[8]:


# 删除空词
def del_blank(sent):
    try:
        while True:
            sent.remove('')
    except:
        return sent


# 用 preprocessing 函数去除句内特殊字符，更新其他项。
# df['sentences'] = df['sentences'].apply(lambda t:[preprocess(sent) for sent in t])
# df['num_sentences'] = df['sentences'].apply(lambda t:len(t))
# 先删除此列节省内存
del df['words']
df['words'] = df['sentences'].apply(lambda t: [del_blank(word_tokenize(preprocess(sent))) for sent in t])
df['num_words'] = df['words'].apply(lambda t: sum([len(sent) for sent in t]))
df

# ### 得到词库

# In[9]:


# 得到词汇表，防止在后续过程中出现未知单词
vocab = set()
for text in df['words']:
    for sent in text:
        vocab.update(set(sent))
vocab.update({'<end>','<begin>','<end>'})
print(len(vocab))

# ## Task-2

# ### 将数据集随机分为90%训练集和10%测试集

# In[15]:


# 设置 90% 的训练集
np.random.seed(0)
train_rate = 0.9
arr = np.arange(len(df))

# 随机打乱 arr 顺序，以达到打乱训练集和测试集的效果
np.random.shuffle(arr)
train_set = df['words'].iloc[arr[:int(len(arr) * train_rate)]]
test_set = df['words'].iloc[arr[int(len(arr) * train_rate):]]
print("Length of train_set:", len(train_set))
print("Length of test_set:", len(test_set))


# ### 定义 n-gram 语言模型
# 在 ngrams_model 类中，定义了 add-one 和 kenser-ney 两种平滑函数

# In[11]:


class ngrams_model:
    def __init__(self, n, method, vocab):
        self.n = n
        # 预定义 self.dic 中的每个 value 均为 ConditionalFreqDist 类， self.dic 第一层选择 n-gram（应对需要回退或者插值的情况），
        # 第二层选择 context，第三层选择 word
        self.prefix = defaultdict(ConditionalFreqDist)
        # unigram 不需要 context
        self.uni = FreqDist()
        self.vocab = vocab
        if method in ['add_one', 'kneser_ney']:
            self.method = method
        else:
            raise 'Method is restricted in `Add-one` or `Kneser-Ney`'

    @staticmethod
    def pad(sent, k):
        if k==1:
            return sent+['<end>']
        pad1 = ['_'] * (k - 1) + ['<begin>']
        pad2 = ['<end>']
        return pad1 + sent + pad2

    def fit(self, dataset):
        # 得到各个长度的语言模型上的 context 和 word 的对应关系
        if self.method == 'add_one':
            k = self.n
            for text in dataset:
                for sent in text:
                    # 在每句话前后加 padding
                    sent = ngrams_model.pad(sent, k)
                    for i in range(len(sent) - k + 1):
                        gram = tuple(sent[i:i + k])
                        if k == 1:
                            self.uni[gram[0]] += 1
                        else:
                            self.prefix[k][gram[:-1]][gram[-1]] += 1
        elif self.method == 'kneser_ney':
            self.suffix = defaultdict(ConditionalFreqDist)
            #self.lam_q = defaultdict(ConditionalFreqDist)
            self.total_gram=defaultdict(int)
            self.middle=defaultdict(ConditionalFreqDist)
            for k in range(1, self.n + 2):
                for text in dataset:
                    for sent in text:
                        # 在每句话前后加 padding
                        sent = ngrams_model.pad(sent, k)
                        for i in range(len(sent) - k + 1):
                            gram = tuple(sent[i:i + k])
                            if k == 1:
                                self.uni[gram[0]] += 1
                            else:
                                self.prefix[k][gram[:-1]][gram[-1]] += 1
                                self.suffix[k][gram[1:]][gram[0]] += 1
                            if k>=3:
                                self.middle[k][gram[1:-1]][(gram[0],gram[-1])]=1
                                #self.lam_q[k] = sum(1 for w in self.prefix[k][context] if self.suffix[n + 1][tuple(list(context) + [w])].N())
                self.total_gram[k] = sum(dic.B() for dic in self.prefix[k].values())
            #self.total_bigram = len(self.vocab)**2

    def add_one(self, context, word):
        n = len(context) + 1
        if n == 1:
            dic = self.uni
        else:
            dic = self.prefix[n][context]
        # 进行 add-1 平滑
        up = dic[word] + 1
        down = dic.N() + len(self.vocab)
        return up / down

    def kneser_ney(self, context, word, d=0.1):
        n = len(context) + 1
        if n == 1:
            # P_continuation()
            up = self.suffix[2][(word,)].B()
            down = self.total_gram[2]
            res = (up + 1) / (down + len(self.vocab))
            return res
        if n == self.n:
            dic = self.prefix[n][context]
            if dic[word]:
                a = max(0, dic[word] - d) / dic.N()
            else:
                a = 0
            if dic.B() == 0:
                lam = 1
            else:
                lam = d * dic.B() / dic.N()
        else:
            # 简化后续代码
            # Continuation count(x)= Number of unique single word contexts for x
            context_plus = tuple(list(context) + [word])
            c_kn_up = self.suffix[n + 1][context_plus].B()
            c_kn_down = self.total_gram[n+2]
            if c_kn_up:
                a = max(0, c_kn_up - d) / c_kn_down
            else:
                a = 0
            q = self.middle[n+1][context[1:]].N()
            #q_ = sum(1 for w in self.prefix[n][context] if self.suffix[n + 1][tuple(list(context) + [w])].B())
            #print(q,q_)
            if q == 0:
                lam = 1
            else:
                lam = d*q/c_kn_down
        # 计算 lambda


        return a + lam * self.kneser_ney(context[1:], word, d)  # 递归

    def perplexity(self, sent):
        # 计算困惑度
        # 如果输入为 string，对其进行 preprocessing 以及分词
        if isinstance(sent, str):
            sent = word_tokenize(preprocess(sent))
        sent_ = ngrams_model.pad(sent, self.n)
        ngrams_ = [tuple(sent_[i:i + self.n]) for i in range(len(sent_) - self.n + 1)]
        # 计算 log probability，防止数据下溢
        log_prob = -sum([log(getattr(self, self.method)(gram[:-1], gram[-1]), 2) for gram in ngrams_]) / len(sent)
        return pow(log_prob, 2)  # 由 log probabilitu 转换回 probability

    def test(self, dataset):
        # 计算测试集各句子平均困惑度
        res = []
        for text in dataset:
            for sent in text:
                res.append(self.perplexity(sent))
        return sum(res) / len(res)

    def generate(self, early_stop=None):
        # 生成句子
        # 设置初始 context
        context = tuple(['_'] * (self.n - 2) + ['<begin>'])[:self.n - 1]
        gram = '<begin>'
        sent = []
        keys = list(self.vocab)
        i = 0
        while gram != '<end>':
            # 获得一个 [0,1] 的概率
            p = np.random.rand()
            # 获得词汇表中所有词的概率 pdf
            pdf = []
            for key in keys:
                pdf.append(getattr(self, self.method)(context, key))
            # 累加得到 cdf
            cdf = np.array(pdf).cumsum()
            # 找到随机概率对应的 word
            gram = keys[cdf.searchsorted(p)]
            #print(gram, end=' ')
            sent.append(gram)
            context = tuple(list(context[1:]) + [gram])
            i += 1
            # 是否需要提前停止
            if early_stop is not None and i >= early_stop:
                sent.append('<end>')
                break
        # 去除最后的 <end> 
        sent = sent[:-1]
        return sent

    def __call__(self, *args, **kwargs):
        # 可直接调用模型计算单词概率
        return getattr(self, self.method)(*args, **kwargs)


# ### 函数验证概率和是否为 1

# In[12]:


def prob_sum(model):
    # 随意选择的 context
    context = tuple(['_'] * (model.n - 2) + ['<begin>'])[:model.n - 1]
    l = []
    for key in model.vocab:
        l.append(model(context, key))
    print('%d-gram language model using %s smoothing: P(w|context) add up to' % (model.n, model.method), sum(l))
    # return l



# ### 得到 unigram, bigram, trigram 语言模型，使用 Add-one 或 Kneser-Ney 两种平滑函数

# In[13]:


# 将结果保存子 rusults 字典中
results = defaultdict(dict)
for method in ['add_one','kneser_ney']:
    for n in [1,2,3]:
        model = ngrams_model(n, method, vocab)
        model.fit(train_set)
        # 计算模型在测试集上的困惑度
        results[(method,n)]['test_perplexity']=model.test(test_set)
        # 使用模型生成 5 个句子
        prob_sum(model)
        results[(method, n)]['generate_sentences'] = [model.generate(50) for i in range(5)]
        # prob_sum(model)
        # 删除 model 释放内存
        model
        # break
    # break

# ### 查看各 model 在测试集所有句子上的平均困惑度

# In[ ]:


frame = pd.DataFrame(
    [results[(method, n)]['test_perplexity'] for method in ['add_one', 'kneser_ney'] for n in [1, 2, 3]],
    index=[(method, n) for method in ['add_one', 'kneser_ney'] for n in [1, 2, 3]], columns=['Perplexity'])
frame.columns.name = ('method', 'n-gram')
frame

# ### 查看各 model 生成的句子

# In[ ]:


for method in ['add_one', 'kneser_ney']:
    for n in [1, 2, 3]:
        print('%d-gram language model using %s smoothing.' % (n, method))
        for i in range(5):
            print('\t%s' % results[(method, n)]['generate_sentences'][i])
        print()


# ## Task-3

# ### 定义 NaiveBayes 模型
# 在 NaiveBayes 类中，实现了计算任意输入 label 的概率，任意输入词相对于任意输入 label 的条件概率，以及在测试集上的评估

# In[ ]:


class NaiveBayes:
    def __init__(self, vocab, alpha=1):
        # 定义 self.dic 为 ConditionalFreqDist 类，第一层选择 label，第二层选择 word
        self.dic = ConditionalFreqDist()
        # 定义 self.label_freq 为 FreqDist 类, 用于统计 label 词频
        self.label_freq = FreqDist()
        # Laplace 平滑系数
        self.alpha = alpha

    def fit(self, train_set):
        # 统计各条件概率和先验概率
        for text_idx in train_set.index:
            label = train_set.loc[text_idx, 'label']
            self.label_freq[label] += 1
            for sent in train_set.loc[text_idx, 'words']:
                for word in sent:
                    self.dic[label][word] += 1

    def calculate(self, label, word=None):
        # 计算单个条件概率或先验概率
        if word is None:
            # P(label)，不进行平滑
            up = self.label_freq[label]
            down = self.label_freq.N()
            return up / down
        else:
            # P(word|label)，进行 Laplace 平滑
            up = self.dic[label][word]
            down = self.dic[label].N()
            return (up + self.alpha) / (down + self.alpha * len(vocab))

    def __call__(self, text):
        # 计算给定文本为各个 label 的概率
        # 假如输入数据不符合格式，对其进行处理
        if isinstance(text, str):
            text = word_tokenize(preprocess(text))
        elif isinstance(text[0], list):
            text = [word for sent in text for word in sent]
        res = {}
        for label in self.label_freq.keys():
            # 防止数字下溢，使用 log probability
            log_p = log(self.calculate(label), 2)
            for word in text:
                log_p += log(self.calculate(label, word), 2)
            res[label] = log_p
        # 对 log probability 进行排序，最大可能性的 label 即为预测结果
        log_sorted = sorted(res.items(), reverse=True, key=lambda x: x[1])
        pred = log_sorted[0][0]
        return {'prediction': pred, 'log_prob': log_sorted}

    def test(self, test_set, beta=1):
        # 在验证集上进行评估
        # 总 label 数
        n = self.label_freq.B()
        # label 列表
        labels = list(self.label_freq.keys())
        # 测试集文本数
        sum_ = len(test_set.index)
        # 总混淆矩阵
        confusion_matrix = pd.DataFrame(np.zeros((n, n)), index=labels, columns=labels)
        # 各类别的混淆矩阵
        confusion_matrix_each_class = dict()
        # 统计每次预测结果，记录于总混淆矩阵
        for text_idx in test_set.index:
            label = test_set.loc[text_idx, 'label']
            pred = self.__call__(test_set.loc[text_idx, 'words'])['prediction']
            confusion_matrix.loc[pred, label] += 1
        confusion_matrix = confusion_matrix.astype('int64')
        # 计算各类别混淆矩阵
        for label in labels:
            tp = confusion_matrix.loc[label, label]
            fp = confusion_matrix.loc[label].sum() - tp
            fn = confusion_matrix.loc[:, label].sum() - tp
            tn = sum_ - tp - fp - fn
            confusion_matrix_each_class[label] = pd.DataFrame([[tp, fp], [fn, tn]], index=['+', '-'],
                                                              columns=['+', '-']).astype('int64')

        # 防止出现除以 0 出现 nan
        def na2zero(x):
            if np.isnan(x):
                return 1
            return x

        # macro precision，recall
        precision_macro = sum(na2zero(matrix.loc['+', '+'] / matrix.loc['+', :].sum()) for matrix in
                              confusion_matrix_each_class.values()) / n
        recall_macro = sum(na2zero(matrix.loc['+', '+'] / matrix.loc[:, '+'].sum()) for matrix in
                           confusion_matrix_each_class.values()) / n
        confusion_matrix_micro = sum(confusion_matrix_each_class.values())
        # micro precision，recall
        pecision_micro = na2zero(confusion_matrix_micro.loc['+', '+'] / confusion_matrix_micro.loc['+', :].sum())
        recall_micro = na2zero(confusion_matrix_micro.loc['+', '+'] / confusion_matrix_micro.loc[:, '+'].sum())
        # 防止 precision,recall 均为0，无法计算 F1
        if precision_macro == 0 and recall_macro == 0:
            F1_macro = 0
        else:
            F1_macro = (1 + beta ** 2) * precision_macro * recall_macro / (beta ** 2 * precision_macro + recall_macro)
        if pecision_micro == 0 and recall_micro == 0:
            F1_micro = 0
        else:
            F1_micro = (1 + beta ** 2) * pecision_micro * recall_micro / (beta ** 2 * pecision_micro + recall_micro)
        return {
            'precision_macro': precision_macro, 'recall_macro': recall_macro,
            'precision_micro': pecision_micro, 'recall_micro': recall_micro,
            'F1_macro': F1_macro, 'F1_micro': F1_micro,
            'confusion_matrix': confusion_matrix, 'confusion_matrix_each_class': confusion_matrix_each_class,
            'confusion_matrix_micro': confusion_matrix_micro
        }


# ### 将数据集分为训练集和测试集（10%），训练集有 30%、50%、70%、90% 四个版本

# In[ ]:


train_rate = [0.3, 0.5, 0.7, 0.9]
test_rate = 0.1
np.random.seed(0)
arr = np.arange(len(df))
np.random.shuffle(arr)
train_set = {}
for rate in train_rate:
    train_set[rate] = df[['label', 'words']].iloc[arr[:int(len(arr) * rate)]]
    print("Length of train_set %.2f:" % rate, len(train_set[rate]))
test_set = df[['label', 'words']].iloc[arr[-int(len(arr) * test_rate):]]
print("Length of test_set %.2f:" % test_rate, len(test_set))

# ### 用 NaiveBayes 模型在四种大小的训练集上训练，并用测试集测试
# 由于这部分占内存较小，所以不删除模型

# In[ ]:


models_NB = defaultdict(dict)
for rate in train_rate:
    models_NB[rate]['model'] = NaiveBayes(vocab)
    # 模型在训练集上训练
    models_NB[rate]['model'].fit(train_set[rate])
    # 在测试集上进行评估
    models_NB[rate]['test_results'] = models_NB[rate]['model'].test(test_set)

# ### 输出在测试集上的评估结果

# In[ ]:


i = 0
for rate in train_rate:
    i += 1
    print('%d.use %2.f%% of documents:\n' % (i, rate * 100))
    print('macro precision: %.2f,\tmacro recall: %.2f,\tmacro F1-score: %.2f' % (
    models_NB[rate]['test_results']['precision_macro'], models_NB[rate]['test_results']['recall_macro'],
    models_NB[rate]['test_results']['F1_macro']))
    print('micro precision: %.2f,\tmicro recall: %.2f,\tmicro F1-score: %.2f' % (
    models_NB[rate]['test_results']['precision_micro'], models_NB[rate]['test_results']['recall_micro'],
    models_NB[rate]['test_results']['F1_micro']))
    print('\nConfusion matrix:\n')
    display(models_NB[rate]['test_results']['confusion_matrix'])
    print('\n\n')

# In[ ]:
time2=time.time()
print(time2-time1)