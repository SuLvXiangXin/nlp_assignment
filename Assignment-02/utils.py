import nltk
from nltk import sent_tokenize, word_tokenize, ngrams, FreqDist, ConditionalFreqDist
from collections import defaultdict
import numpy as np
import json
import re
import random
import math
import time
import pickle
import pandas as pd
import copy
import os
from IPython.display import display


def preprocess(sentence):
    # 将句内除字母、数字、空格外的所有字符替换为空格
    res = re.sub(r'[^\w\s]', ' ', sentence)
    return res.lower()


class dataset:
    def __init__(self, words, **kwargs):
        # 对数据集进行采样，随机丢弃出现频率高的单词
        self.words = dataset.downsample(words)
        self.vocab = FreqDist([word for text in self.words for word in text])
        self.vocab['<unknown>'] = 0
        # 序号->单词 映射，频率
        self.index2word, self.frequency = np.array(list(self.vocab.items())).T
        self.frequency = self.frequency.astype(np.float32)
        self.frequency /= self.frequency.sum()
        self.indexes = np.arange(len(self.index2word))
        # 单词->序号 映射
        self.word2index = {word: i for i, word in enumerate(self.index2word)}
        # 获得中心词，上下文，负采样
        self.centers, self.contexts, self.negatives = self.get_center_context(**kwargs)

    @staticmethod
    def downsample(words, t=1e-4):
        '''
        下采样高频词
        '''
        vocab = FreqDist([word for sent in words for word in sent])
        N = vocab.N()

        # 按照概率，随机丢弃高频词
        def drop(word):
            return random.random() < max(1 - math.sqrt(t / (vocab[word] / N)), 0)

        return [[word for word in text if not drop(word)] for text in words]

    def get_center_context(self, **kwargs):
        """
        获得中心词、上下文、负采样
        """
        window = kwargs.get('window', 2)
        alpha = kwargs.get('alpha', 0.75)
        k = kwargs.get('k', 5)
        centers = []
        contexts = []
        # 从数据集中获得中心词和上下文
        for sent in self.words:
            for i in range(window, len(sent) - window):
                center = self.word2index[sent[i]]
                centers.append(center)
                context = sent[i - window:i] + sent[i + 1:i + window + 1]
                contexts.append([self.word2index[word] for word in context])
        centers = np.array(centers)
        contexts = np.array(contexts)
        weights = np.array([self.vocab[word] for word in self.index2word]) ** alpha
        weights = weights / np.sum(weights)
        # 负采样
        negatives = self.negative_sample(contexts, weights, k)
        not_na = (negatives.isna().sum(axis=1) == 0).values
        negatives = negatives.values
        return centers[not_na], contexts[not_na], negatives[not_na].astype(int)

    def negative_sample(self, contexts, weights, k):
        """
        负采样
        """
        # 随机生成负采样样本
        neg_sample = np.random.choice(self.indexes, size=(len(contexts), len(contexts[0]) * k), p=weights)
        a = contexts[:, None].repeat(len(contexts[0]) * k, axis=1)
        b = neg_sample[:, :, None]
        # 获得与上下文有重叠的部分（需要被取代）
        condition = (~(a == b)).prod(axis=-1)
        new = np.where(condition, neg_sample, np.nan)
        df = pd.DataFrame(new)
        # 用前项或后项填充重叠部分
        df.fillna(method='bfill', axis=1, inplace=True)
        df.fillna(method='ffill', axis=1, inplace=True)
        return df

    def __getitem__(self, i):
        """
        从数据集中获取中心词、上下文、负采样
        """
        centers = self.centers[i]
        context_neg = np.concatenate([self.contexts[i], self.negatives[i]], axis=1)
        # 上下文标签为 1，负采样标签为 0
        labels = np.concatenate([np.ones_like(self.contexts[i]), np.zeros_like(self.negatives[i])], axis=1)
        return centers, context_neg, labels

    def __len__(self):
        """
        数据集大小
        """
        return len(self.centers)


class batch_loader:
    def __init__(self, dataset, batch=16, shuffle=True):
        self.dataset = dataset
        self.batch = batch
        self.i = 0
        self.N = len(dataset)
        # 随机取样
        self.shuffle = shuffle

    def __iter__(self):
        return self

    def __next__(self):
        if self.i > self.N:
            self.i = 0
            # raise StopIteration
        self.i += self.batch
        # 随机取样
        if self.shuffle:
            x = random.randint(1, self.N - self.batch)
        else:
            x = self.i
        return self.dataset[x:x + self.batch]


class mymodel:
    def __init__(self, embed_dim, vocab_size):
        self.dim = embed_dim
        self.vocab_size = vocab_size
        # 中心词矩阵
        self.embed_v = np.random.randn(self.vocab_size, self.dim)
        # 上下文矩阵
        self.embed_w = np.random.randn(self.vocab_size, self.dim)

    def __call__(self, center, context, label):
        self.center = center
        self.context = context
        # 从中心词、上下文矩阵中取出对应的词向量
        self.v = v = np.expand_dims(self.embed_v[center], 1)
        self.w = w = self.embed_w[context]
        # 中心词向量和上下文向量点乘
        pred = (v * w).sum(axis=-1)
        # 计算 sigmoid
        logit = sigmoid(pred)
        # 损失函数
        loss = -np.mean((label * np.log(logit) + (1 - label) * np.log(1 - logit)))
        # 计算中心词和上下文矩阵的梯度，便于后续梯度下降
        self.dv = -(w * np.expand_dims(label * (1 - logit) - (1 - label) * logit, -1)).sum(axis=1, keepdims=True)
        self.dw = -(v * np.expand_dims(label * (1 - logit) - (1 - label) * logit, -1))
        return loss

    def step(self, lr=1e-5):
        """
        反向传播更新参数
        """
        # 梯度下降
        self.embed_v[self.center] -= self.dv.reshape(-1, self.dv.shape[-1]) * lr
        self.embed_w[self.context.reshape(-1)] -= self.dw.reshape(-1, self.dw.shape[-1]) * lr

    def save(self, path):
        np.save(path + '.npy',
                {'dim': self.dim, 'vocab_size': self.vocab_size, 'embed_v': self.embed_v, 'embed_w': self.embed_w})

    def load(self, path):
        dic = np.load(path + '.npy', allow_pickle=True).item()
        self.dim = dic['dim']
        self.vocab_size = dic['vocab_size']
        self.embed_v = dic['embed_v']
        self.embed_w = dic['embed_w']

def sigmoid(x):
    """
    sigmoid 函数
    """
    s = 1 / (1 + np.exp(-x)+1e-8)
    return s

def onehot(x,size):
    """
    将序号转换为onehot形式（未使用，由于计算量太大，效率很低）
    """
    if len(x.shape)==2:
        return (np.arange(size)==x[:,:,None]).astype(int)
    elif len(x.shape)==1:
        return (np.arange(size)==x[:,None]).astype(int)


def train(kwargs):
    """
    训练函数
    """
    # 如果改变 windows，k，alpha 等超参数，则需要更新训练集
    if kwargs.get('change',False):
        train_set.centers,train_set.contexts,train_set.negatives=train_set.get_center_context(**kwargs)
    # 获得 batchloader
    train_loader=batch_loader(train_set,kwargs['batch'],shuffle=kwargs.get('shuffle',False))
    # 获得模型
    m=mymodel(kwargs['embed_dim'],len(train_set.vocab))
    if os.path.exists(kwargs['exp_name']) and kwargs.get('load',False):
        print('loading weights...')
        m.load(kwargs['exp_name'])
    i=0
    lr=kwargs['lr']
    best_loss=10000
    best_iter=0
    for data in train_loader:
        i+=1
        # 前向传播，计算 loss
        loss=m(*data)
        # 反向传播更新参数
        m.step(lr=lr)
        if i%20000==0 and kwargs.get('print',True):
            print('Exp:%s, Iteration:%06d, Loss:%.3f'%(kwargs['exp_name'],i,loss))
        # learning rate decay
        if i%kwargs['lr_decay_step']==0:
            lr/=10
            if kwargs.get('print',True):
                print('lr change to %f'%lr)
        # 保存最佳 loss，用于后续比较
        if best_loss>loss:
            best_loss=np.copy(loss)
            best_iter=i
        if kwargs.get('early_stop',None) and kwargs['early_stop']<i:
            break
    m.save(kwargs['exp_name'])
    print('Exp:%s, best iter:%d, best loss:%f'%(kwargs['exp_name'],best_iter,best_loss))
    return {'loss':loss,'model':m}

def generate_pairs(dataset,n=100):
    """
    随机获取词对
    """
    # 随机获取词对
    c=np.random.choice(dataset.indexes,size=(n,2),p=dataset.frequency)
    # 去除重复词对（两个词相同的词对）
    duplicate=c[:,0]==c[:,1]
    s=duplicate.sum()
    while s:
        a=np.random.choice(dataset.indexes,size=(s,2),p=dataset.frequency)
        c[duplicate]=a
        duplicate=c[:,0]==c[:,1]
        s=duplicate.sum()
    return c

def compute_similarity(pairs,model):
    """
    根据词对计算 cos 相似度
    """
    data=model.embed_v[pairs]
    x=data[:,0]
    y=data[:,1]
    cos=np.abs((x*y).sum(axis=1))/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1))
    return cos

def doc_embed_all_words(text,model,train_set):
    words_idx=np.array([train_set.word2index.get(word,train_set.word2index['<unknown>']) for word in text])
    return model.embed_v[words_idx].mean(axis=0)

# 对文章前n个词的词向量取平均获得文章向量
def doc_embed_first_n(text,model,train_set,n=100):
    words_idx=np.array([train_set.word2index.get(word,train_set.word2index['<unknown>']) for word in text[:n]])
    return model.embed_v[words_idx].mean(axis=0)

def get_all_doc_embed(texts,model,train_set,method=doc_embed_all_words,**kwargs):
    """
    根据之前定义的方法获得文档向量
    """
    embeds=[]
    for text in texts:
        embeds.append(method(text,model,train_set,**kwargs))
    embeds=np.array(embeds)
    return embeds

def k_means(doc_embeds,k=10):
    """
    对所有 document 进行 k-means 聚类
    """
    # 初始化 k 个类中心为所有文档中的任意 k 个
    centers=doc_embeds[np.random.randint(doc_embeds.shape[0],size=k)]
    centers_pre=np.zeros_like(centers)
    embeds=doc_embeds[:,None,:].repeat(k,axis=1)
    df=pd.DataFrame(doc_embeds)
    i=0
    while (centers!=centers_pre).sum()>0:
        i+=1
        centers_pre=np.copy(centers)
        # 对每个文档分类到其距离最近的类中
        classify=np.linalg.norm(embeds-centers,axis=-1).argsort(axis=-1)[:,0]
        # 重新计算类中心
        group_mean=df.groupby(classify).mean()
        index=group_mean.index.values
        values=group_mean.values
        centers[index]=values
        if i%20==0:
            print('Iter:%d, loss:'%i,np.square(centers-centers_pre).sum())
            print('Number of each class:',np.bincount(classify))
    print('\nIt takes %d iterations.'%i)
    print('Number of each class:',np.bincount(classify))
    return classify

def get_confusion(classify,label):
    """
    评估聚类效果，获得 Confusion Matrix
    """
    cu=classify[:,None]==classify[None,:]
    lei=label[:,None]==label[None,:]
    # 减去和自己的比较
    tp=(cu&lei).sum()-len(classify)
    fn=((~cu)&lei).sum()
    fp=(cu&(~lei)).sum()
    tn=((~cu)&(~lei)).sum()
    return pd.DataFrame([[tp,fn],[fp,tn]],index=['同类','非同类'],columns=['同簇','非同簇'])

def micro_f1(classify,label,beta=1):
    """
    计算 micro F1-scroe
    """
    confusion=get_confusion(classify,label)
    precision=confusion.iloc[0,0]/confusion.iloc[:,0].sum()
    recall=confusion.iloc[0,0]/confusion.iloc[0,:].sum()
    f1=(1+beta**2)*(precision*recall)/(beta**2*precision+recall)
    display(confusion)
    print('Precision:%.2f%% ,Recall:%.2f%% ,Micro F1:%.2f%%'%(precision*100,recall*100,f1*100))


def cal_dist(data, n=200):
    """
    根据向量矩阵计算距离矩阵
    """
    N = data.shape[0]
    dist = np.zeros((N, N))
    # 防止爆内存，分批计算
    for i in range(0, N, n):
        dist[i:i + n, :] = np.linalg.norm(data[None, :].repeat(n, axis=0) - data[i:i + n][:, None], axis=-1)
    return dist


def calc_p_and_entropy(dist, beta):
    """
    计算高维向量概率/熵
    """
    n = dist.shape[0]
    p = np.exp(-np.square(dist) * beta[:, None])
    # 防止数字下溢，现将对角线设为 0
    p[range(n), range(n)] = 0
    p_sum = p.sum(axis=1, keepdims=True)
    # 防止取 log 时对角线上为 0
    p[range(n), range(n)] = 1
    p /= p_sum
    # 计算熵
    log_entropy_matrix = -(p * np.log(p))
    log_entropy = log_entropy_matrix.sum(axis=1) - log_entropy_matrix[range(n), range(n)]
    p[range(n), range(n)] = 0
    return p, log_entropy


def binary_search(dist, init_beta, perplexity, threshold=1e-5, max_iter=50):
    """
    二分法搜索最佳 beta 值
    """
    print("寻找最佳 beta...")
    n = dist.shape[0]
    # 初始化 beta 上下限
    beta_max = np.array([np.inf] * n, dtype=np.float32)
    beta_min = np.array([-np.inf] * n, dtype=np.float32)
    beta = np.array([init_beta] * n, dtype=np.float32)
    # 计算高维向量概率/熵
    P, log_entropy = calc_p_and_entropy(dist, beta)
    # 计算与设定困惑度的差值
    diff = log_entropy - perplexity
    i = 0
    while np.abs(diff).max() > threshold and i < max_iter:
        # 更新上下限
        beta_min[diff > 0] = beta[diff > 0]
        beta_max[diff <= 0] = beta[diff <= 0]
        # 交叉熵比期望值大，增大beta
        beta[(diff > 0) & (beta_max == np.inf)] *= 2.
        beta[(diff > 0) & (beta_max != np.inf)] = (beta[(diff > 0) & (beta_max != np.inf)] + beta_max[
            (diff > 0) & (beta_max != np.inf)]) / 2.
        # 交叉熵比期望值小， 减少beta
        beta[(diff <= 0) & (beta_min == -np.inf)] /= 2.
        beta[(diff <= 0) & (beta_min != -np.inf)] = (beta[(diff <= 0) & (beta_min != -np.inf)] + beta_min[
            (diff <= 0) & (beta_min != -np.inf)]) / 2.
        # 重新计算
        p, log_entropy = calc_p_and_entropy(dist, beta)
        diff = log_entropy - perplexity
        print('iter %d' % (i + 1), ',max difference of log-entropy:%.6f' % np.abs(diff).max())
        i += 1
        # 返回最优的 beta 以及所对应的 P
    return p, beta


def p_joint(data, init_beta=1, perplexity=5):
    """
    计算高维联合概率
    """
    N = data.shape[0]
    # 计算距离
    if os.path.exists('dist.pkl'):
        print('loading dist...')
        with open('dist.pkl','rb') as f:
            dist = pickle.load(f)
    else:
        print('dumping dist...')
        dist = cal_dist(data, n=1000)
        with open('dist.pkl','wb') as f:
            pickle.dump(dist,f)
    # 二分法获得最佳 beta
    p, beta = binary_search(dist, init_beta, perplexity)
    p_join = (p + p.T) / 2
    p_join /= p_join.sum()
    p_join[range(N), range(N)] = 1
    print("Mean value of beta: %f" % np.mean(beta))
    return p_join


def q_tsne(dist):
    """
    计算低维联合概率：t分布
    """
    N = dist.shape[0]
    tmp = (1 + np.square(dist)) ** -1
    tmp[range(N), range(N)] = 0
    # 归一化
    q = tmp / tmp.sum()
    # 设对角线为非 0 值，方便后面计算
    q[range(N), range(N)] = 1
    return q


def draw_pic(data, labs, name='1.jpg'):
    """
    画图
    """
    plt.cla()
    unque_labs = np.unique(labs)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unque_labs))]
    p = []
    legends = []
    for i in range(len(unque_labs)):
        index = np.where(labs == unque_labs[i])
        pi = plt.scatter(data[index, 0], data[index, 1], c=[colors[i]])
        p.append(pi)
        legends.append(unque_labs[i])

    plt.legend(p, legends)
    # plt.savefig(name)
    plt.show()


def tsen(data, dim, init_beta, target_perplexity, plot=False):
    """
    计算 tsne
    data:文档向量
    dim:低维向量维度
    init_beta:初始化beta值
    target_perplexity:目标困惑度
    """
    N, D = data.shape
    # 随机初始化低维数据
    y = np.random.randn(N, dim)
    # 计算高维向量的联合概率
    print("1.计算高维向量的联合概率")
    p = p_joint(data, init_beta, target_perplexity)
    # 开始进行迭代训练
    # 训练相关参数，用 Adam 算法迭代
    print("2.迭代计算低维向量的联合概率")
    max_iter = 1000
    lr = 3000
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = np.zeros_like(y)
    v = np.zeros_like(y)
    for m_iter in range(max_iter):
        # 低维距离
        dist_y = cal_dist(y)
        # 低维联合概率
        q = q_tsne(dist_y)
        # 计算梯度
        y_minus = y[:, None].repeat(N, axis=1) - y[None, :].repeat(N, axis=0)
        dy = 4 * ((p - q)[:, :, None] * y_minus * (1 + dist_y ** 2)[:, :, None] ** -1).sum(axis=1)
        # Adam 优化器
        # m=(1-beta1)*m+beta1*dy
        # v=(1-beta2)*v+beta2*dy**2
        # m_hat=m/(1-beta1**(m_iter+1))
        # v_hat=v/(1-beta2**(m_iter+1))
        # y-=lr*m_hat/(np.sqrt(v_hat)+eps)
        y -= lr * dy
        # 损失函数
        if (m_iter + 1) % 1 == 0:
            c = p * np.log(p / q)
            loss = c.sum() - c[range(N), range(N)].sum()
            print("Iteration %d: ,loss: %f" % (m_iter + 1, loss))
        if plot and m_iter % 100 == 0:
            print("Draw Map")
            draw_pic(Y, labs, name="%d.jpg" % (m_iter))

    return y



