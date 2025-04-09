# 文本特征提取方法说明
## 一、核心功能说明
代码的核心功能是从一个文本列表中提取特征，使用两种不同的方法：词频（CountVectorizer）和TF-IDF（TfidfVectorizer）。这两种方法都是文本特征提取中常用的技术，用于将文本数据转换为机器学习模型可以处理的数值形式。
### 1. 高频词模式（high_frequency）
- ​**原理**：统计每个词在所有文档中的出现次数
- ​**实现**：使用 `CountVectorizer`
- ​**公式**：
词频（Term Frequency, TF）：
$$\[ TF(t, d) = \frac{n_t}{N} \]$$
  分子：在某一类中词条出现的次数
  分母：该类中所有的词条数目
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
```
运行结果
<img width="890" alt="3 1" src="https://github.com/user-attachments/assets/7ca191b2-592a-4c51-b5c3-7775b66976fd" />

# 附加5
```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings

# 忽略弃用警告（临时解决方案，建议尽快升级库）
warnings.filterwarnings("ignore", category=FutureWarning)

# ======================
# 1. 数据生成（增强可复现性）
# ======================
np.random.seed(42)  # 固定随机种子
n_samples = 151
n_features = 10

# 生成特征和标签
X = np.random.rand(n_samples, n_features)
y = np.concatenate([np.ones(127, dtype=int), np.zeros(24, dtype=int)])

# 转换为 DataFrame 并保存原始索引
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
data['label'] = y
data.index.name = 'sample_id'  # 添加索引列名

# ======================
# 2. 数据划分（明确参数）
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns='label'),
    data['label'],
    test_size=0.2,
    random_state=42,
    stratify=data['label']  # 保持类别比例
)

# ======================
# 3. 类别平衡处理（可视化验证）
# ======================
print("原始数据分布：")
print(data['label'].value_counts())

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("过采样后训练集分布：")
print(pd.Series(y_resampled).value_counts())

# ======================
# 4. 模型训练（优化参数）
# ======================
model = RandomForestClassifier(
    n_estimators=100,           # 显式设置树的数量
    class_weight='balanced',    # 自动调整类别权重
    random_state=42
)
model.fit(X_resampled, y_resampled)

# ======================
# 5. 模型评估（增加可视化）
# ======================
y_pred = model.predict(X_test)

print("分类报告：")
print(classification_report(y_test, y_pred))

print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))

# ======================
# 6. 交叉验证（增强稳定性验证）
# ======================
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model,
    X_resampled,
    y_resampled,
    cv=5,
    scoring='f1'
)
print(f"5折交叉验证 F1 分数: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
```

运行结果

<img width="620" alt="5 3" src="https://github.com/user-attachments/assets/4373f8c1-1ca3-4fe3-83f2-cdc1d7bf8d1f" />

# 附加6
```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings

# 忽略未来警告（scikit-learn 1.6+ 弃用提示）
warnings.filterwarnings("ignore", category=FutureWarning)

# ======================
# 1. 生成模拟数据（示例数据）
# ======================
# 创建不平衡数据集（1000样本，20特征，类别比例 1:4）
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.8, 0.2],  # 类别权重（多数类:少数类）
    random_state=42
)

# 转换为 DataFrame 方便查看
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
data['label'] = y
print("原始数据分布：")
print(data['label'].value_counts())

# ======================
# 2. 数据划分（训练集/测试集）
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns='label'),
    data['label'],
    test_size=0.3,
    random_state=42,
    stratify=data['label']  # 保持类别比例
)

# ======================
# 3. 处理类别不平衡（SMOTE过采样）
# ======================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("过采样后训练集分布：")
print(pd.Series(y_resampled).value_counts())

# ======================
# 4. 训练分类模型（随机森林）
# ======================
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # 自动调整类别权重
    random_state=42
)
model.fit(X_resampled, y_resampled)

# ======================
# 5. 预测与评估
# ======================
y_pred = model.predict(X_test)

# 输出分类报告（含精度/召回率/F1值）
print("=== 分类评估报告 ===")
print(classification_report(
    y_test,
    y_pred,
    target_names=['普通类', '重要类'],  # 自定义类别名称
    digits=4  # 小数位数
))

# 可视化混淆矩阵
print("=== 混淆矩阵 ===")
conf_matrix = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(
    conf_matrix,
    index=['普通类', '重要类'],
    columns=['预测普通类', '预测重要类']
))
```

运行结果

<img width="536" alt="6 4" src="https://github.com/user-attachments/assets/e539449b-8c31-4583-8f2b-81ad4e3bcd2b" />





