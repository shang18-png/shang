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