from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

np.random.seed(21)
FILENAME = './编码数据1.xlsx'
Data = pd.read_excel(
    FILENAME, header=0, usecols="AM,AN,AO,AP,AQ, BB,BC,BD, BP,BQ,BR, CC,CD,CE,  AT,BG,BU,CH")
DataArray = Data.values
x = DataArray[:, 0:14]
x = np.array(x)
y = DataArray[:, 14:17]
y_sum = np.sum(y == 1, axis=1)  # 输出变量整合成一个，只要出现了两个1(是)就输出1
y = (y_sum >= 2).astype(int)
y = y[:, np.newaxis]

test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, random_state=42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def ReLU(x):
    return np.maximum(0, x)


def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)


class OurNeuralNetwork:
    def __init__(self):
        self.weights1 = np.random.normal(scale=np.sqrt(
            2 / 14), size=(14, 8))
        self.biases1 = np.zeros((1, 8))

        self.weights2 = np.random.normal(scale=np.sqrt(
            2 / 7), size=(8, 9))
        self.biases2 = np.zeros((1, 9))

        self.weights_out = np.random.normal(scale=np.sqrt(
            2 / 4), size=(9, 1))
        self.biases_out = np.zeros((1, 1))
        self.loss = []

    # 前向传播学习
    def feedforward(self, X):
        hidden1_activated = ReLU(np.dot(X, self.weights1) + self.biases1)
        hidden2_activated = ReLU(
            np.dot(hidden1_activated, self.weights2) + self.biases2)
        output = sigmoid(
            np.dot(hidden2_activated, self.weights_out) + self.biases_out)
        return output

    def compute_loss(self, predictions, targets):
        N = predictions.shape[0]
        loss = -np.sum(targets * np.log(predictions + 1e-9) +
                       (1 - targets) * np.log(1 - predictions + 1e-9)) / N
        return loss

    def plot_loss(self):
        plt.plot(range(1, len(self.loss) + 1), self.loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('损失函数')
        plt.savefig('损失函数.png')
        plt.show()

    def train(self, X, Y):
        learn_rate = 0.001
        epochs = 800
        self.loss = []

        for epoch in range(epochs):
            # 前向传播
            self.hidden1_activated = ReLU(
                np.dot(X, self.weights1) + self.biases1)
            self.hidden2_activated = ReLU(
                np.dot(self.hidden1_activated, self.weights2) + self.biases2)
            output = sigmoid(np.dot(self.hidden2_activated,
                             self.weights_out) + self.biases_out)

            # 计算损失
            loss = self.compute_loss(output, Y)
            self.loss.append(loss)
            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch}, Loss: {loss}")

            # 反向传播
            # 输出层的误差
            output_error = (output - Y) * sigmoid_derivative(output)
            d_weights_out = np.dot(self.hidden2_activated.T, output_error)
            d_biases_out = np.sum(output_error, axis=0, keepdims=True)

            # 第二个隐藏层的误差
            hidden2_error = np.dot(
                output_error, self.weights_out.T) * ReLU_derivative(self.hidden2_activated)
            d_weights2 = np.dot(self.hidden1_activated.T, hidden2_error)
            d_biases2 = np.sum(hidden2_error, axis=0, keepdims=True)

            # 第一个隐藏层的误差
            hidden1_error = np.dot(
                hidden2_error, self.weights2.T) * ReLU_derivative(self.hidden1_activated)
            d_weights1 = np.dot(X.T, hidden1_error)
            d_biases1 = np.sum(hidden1_error, axis=0, keepdims=True)

            # 更新
            self.weights_out -= learn_rate * d_weights_out
            self.biases_out -= learn_rate * d_biases_out
            self.weights2 -= learn_rate * d_weights2
            self.biases2 -= learn_rate * d_biases2
            self.weights1 -= learn_rate * d_weights1
            self.biases1 -= learn_rate * d_biases1
        return self

    def get_weights(self):
        return self.weights1


# 5折检验
kf = KFold(n_splits=5)
accuracies = []

for train_index, val_index in kf.split(x_train):
    X_train_fold, X_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model = OurNeuralNetwork()
    model.train(X_train_fold, y_train_fold)

    y_pred_fold = model.feedforward(X_val_fold)
    y_pred_fold_binary = (y_pred_fold >= 0.5).astype(int)

    accuracy_fold = accuracy_score(y_val_fold, y_pred_fold_binary)
    accuracies.append(accuracy_fold)

# 计算5折交叉验证的平均准确率
mean_accuracy = np.mean(accuracies)
print("Mean Accuracy:", mean_accuracy)


# 训练数据
network = OurNeuralNetwork()
network.train(x_train, y_train)
outputs_verify = network.feedforward(x_train)
predictions_verify = [np.round(output) for output in outputs_verify]


def importance():
    weights_input_layer = network.get_weights()
    importance_scores = np.sum(np.abs(weights_input_layer), axis=1)
    importance_scores /= np.sum(importance_scores)  # 归一化
    node_names = ['传统产品增加新口味', '传统产品增加新功能', '拓展新的产品种类', '产品包装创新', '产品共创', '拓展线上渠道', '增设线下体验店', '开展快闪活动',
                  '新型健康服务', '新型诊疗服务', '数智技术加强产品质量监控', '专属IP形象打造', '赞助影视节目', '搭建中医药知识图谱']  # 创建节点名称列表
    assert len(node_names) == len(
        importance_scores), "node_names and importance_scores must have the same length"

    # 从大到小排序
    sorted_indices = np.argsort(importance_scores)
    sorted_importance_scores = importance_scores[sorted_indices]
    sorted_node_names = [node_names[i] for i in sorted_indices]
    for score, name in zip(sorted_importance_scores, sorted_node_names):
        print(f"重要性: {score}, 策略: {name}")

    # 条形图
    cmap = cm.get_cmap('coolwarm')
    vmin, vmax = np.min(sorted_importance_scores), np.max(
        sorted_importance_scores)
    normed_scores = (sorted_importance_scores - vmin) / \
        (vmax - vmin)
    colors = cmap(normed_scores)[:, :-1]
    plt.figure(figsize=(14.5, 6))
    plt.tight_layout()

    plt.barh(sorted_node_names, sorted_importance_scores, color=colors)
    plt.title('重要性排序')
    plt.xlabel('重要性得分')
    plt.savefig('重要性排序.png')
    plt.show()


def ROC_AUC():
    outputs_verify = network.feedforward(x_test)
    y_preds = outputs_verify[:, 0]  # 获取预测结果
    fpr, tpr, thresholds = roc_curve(y_test, y_preds)
    accuracy = accuracy_score(y_test, np.round(y_preds))
    print("Accuracy: %.2f" % accuracy)
    auc_value = roc_auc_score(y_test, y_preds)
    print("AUC值：", auc_value)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.legend()
    plt.savefig('ROC曲线.png')
    plt.show()


# 标题显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

network.plot_loss()
ROC_AUC()
importance()
