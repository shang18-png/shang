<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
# 文本特征提取方法说明
## 一、核心功能说明
代码的核心功能是从一个文本列表中提取特征，使用两种不同的方法：词频（CountVectorizer）和TF-IDF（TfidfVectorizer）。这两种方法都是文本特征提取中常用的技术，用于将文本数据转换为机器学习模型可以处理的数值形式。
### 1. 高频词模式（high_frequency）
- ​**原理**：统计每个词在所有文档中的出现次数
- ​**实现**：使用 `CountVectorizer`
- ​**公式**：
词频（Term Frequency, TF）：
$$\[ TF(t, d) = \frac{n_t}{N} \]$$
-  分子：在某一类中词条出现的次数
-  分母：该类中所有的词条数目
### 2. TF-IDF模式（tfidf）
TF-IDF（term frequency–inverse document frequency，词频-逆向文件频率）是一种用于信息检索（information retrieval）与文本挖掘（text mining）的常用加权技术。

TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

其主要思想是：如果某个单词在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。
- ​**原理**：同时考虑词频（TF）和逆文档频率（IDF）
- ​**实现**：使用 `TfidfVectorizer`
- ​**公式**：  
$$\[ TF-IDF = TF \times IDF \]$$

---

# 文本特征提取方法：高频词与TF-IDF切换指南

## 一、核心功能对比
| 特征模式       | 实现类               | 核心公式                                 | 特点说明                     |
|----------------|----------------------|----------------------------------------|----------------------------|
| 高频词统计     | `CountVectorizer`   | `Frequency(t,d) = 词t在文档d的出现次数` | 简单词频统计，无权重调整     |
| TF-IDF加权     | `TfidfVectorizer`   | `TF-IDF = TF × IDF`                    | 突出重要词汇，抑制常见词     |

---

## 二、代码实现结构
```python
def extract_features(texts, method='high_frequency', ​**kwargs):
    # 方法路由选择
    if method == 'high_frequency':
        vectorizer = CountVectorizer(**kwargs)
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(**kwargs)
    else:
        raise ValueError("Invalid method...")
    
    # 统一处理流程
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

# 自定义参数配置模板
features, vectorizer = extract_features(
    texts,
    method='tfidf',
    stop_words=['the', 'is'],      # 自定义停用词
    ngram_range=(1,2),             # 捕捉1-2词组合
    max_df=0.85,                   # 忽略文档频率>85%的词
    min_df=2                       # 忽略文档频率<2的词
)