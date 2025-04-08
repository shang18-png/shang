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