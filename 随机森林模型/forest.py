import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

np.random.seed(0)
FILENAME = './编码数据.xlsx'
Data = pd.read_excel(
    FILENAME, header=0, usecols="CV,CW,CX")
Data = Data.dropna()
data = Data.values


def elbow(data):
    inertias = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=500, n_init=10, random_state=0)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


def k_means():
    # elbow(data)

    k = 2
    max_iter = 500  # 迭代次数500
    kmeans = KMeans(n_clusters=k, max_iter=max_iter, n_init=10)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 聚类中心的平均值低于3.5则为悲观者，否则为乐观者
    categories = np.array([0 if np.mean(centroids[label])
                          < 3.5 else 1 for label in labels])

    # # 打印聚类中心和类别变量
    # print("Cluster Centers:")
    # print(centroids)
    # print("\nCategories:")
    # print(categories)

    # # 输出每个类别的数量
    # print("\nNumber of pessimists (category 1):", np.sum(categories == 1))
    # print("Number of optimists (category 2):", np.sum(categories == 2))
    return categories


categories = k_means()
print(categories)

df = pd.read_excel(
    FILENAME, header=0, usecols="CY,CZ,DA,DB,DC,DD,DE, DH")
df.dropna(inplace=True)

# 1 确定自变量和因变量
x = df.values[:, :6]  # 自变量
y = categories  # 因变量

# 2 对年龄进行分类
age = df.values[:, 7]


def classify_age(age):
    if age <= 4:
        return '35岁及以下'
    else:
        return '35岁以上'


variable_mapping = {'q27a1': '品质传承', 'q27a2': '文化传承', 'q27a3': '宣传工作',
                    'q27a4': '品牌特色', 'q27a5': '品牌形象', 'q27a6': '个性化服务', 'q27a7': '消费体验'}
df['age_group'] = df['q30'].apply(classify_age)
age_groups = df['age_group'].unique()

# 3 针对不同年龄的群体，分别建立随机森林模型
feature_importances = {}

for age_group in age_groups:
    age_data = df[df['age_group'] == age_group]

    X_age = age_data.iloc[:, :7]
    y_age = categories[age_data.index]

    # 拆分数据集，28分
    X_train, X_test, y_train, y_test = train_test_split(
        X_age, y_age, test_size=0.2, random_state=42)

    # 建立随机森林模型
    rf = RandomForestClassifier(bootstrap=True, random_state=42)

    # 5折交叉验证
    cv_scores = cross_val_score(rf, x, y, cv=5)
    print("交叉验证得分:", cv_scores)
    print("平均准确率:", np.mean(cv_scores))
    print("准确率标准差:", np.std(cv_scores))
    rf.fit(X_train, y_train)

    # 变量重要程度降序排列
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 新映射
    feature_importances[age_group] = {
        variable_mapping[df.columns[i]]: importances[i] for i in indices}

indicator_means_age = df[df['age_group'] == age_group].iloc[:, :7].mean()

# 输出每个年龄组的变量重要程度
for age_group, importance in feature_importances.items():
    print(f"年龄组 {age_group}:")
    for feature, importance_score in importance.items():
        print(f"{feature}: {importance_score}")
    print("\n")

# 4 绘制四分图
matplotlib.rcParams['font.family'] = 'SimHei'
for age_group, importance in feature_importances.items():
    plt.figure(figsize=(8, 6))
    # 计算均值
    indicator_means_age = df[df['age_group'] == age_group].iloc[:, :7].mean()
    importance_medians_age = np.mean(list(importance.values()))

    # 划分象限
    x_threshold = np.median(indicator_means_age)
    y_threshold = np.median(importance_medians_age)

    # 散点图
    x_values = indicator_means_age.values
    y_values = list(importance.values())
    labels = indicators = list(importance.keys())

    plt.scatter(x_values, y_values, label=age_group)

    # 添加变量名称
    for j, txt in enumerate(labels):
        plt.annotate(txt, (x_values[j], y_values[j]), fontsize=8)

    # 阈值线
    plt.axvline(x=x_threshold, color='r', linestyle='--')
    plt.axhline(y=y_threshold, color='r', linestyle='--')

    plt.xlabel('平均满意度')
    plt.ylabel('变量重要性')
    plt.title(f'四分图分析 ({age_group})')
    plt.legend()
    plt.savefig(f'四分图分析_{age_group}.png')
    plt.show()

# ROC-AUC
y_pred_proba = rf.predict_proba(X_test)[:, 1]  # 获取正类别的预测概率值
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线')
plt.legend()
plt.savefig('ROC曲线.png')
plt.show()
auc_score = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC值:", auc_score)
