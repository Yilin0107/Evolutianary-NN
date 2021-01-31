from common.load_data import mnsitDataset, cifarDataset
from common.utils import *
from common.net import bp


# 定义模型
model = bp([28, 28])
dataset = mnsitDataset
weights_path = 'model/bp_mnist-02_step2.npz'
# 加载权重
index, weights_shape = findShape(model.weights)
weights = np.load(weights_path)['pareto'][7, ...]
model.set_weights(setShape(weights, index, weights_shape))
# 加载数据
outputs, targets = forward(model, dataset, batch_size=10000, train=False)
# 计算相关指标
precision, recall = evalute(targets, outputs)
accuracy = accuracy_score(targets, np.argmax(outputs, axis=1))
print('accuracy: {} precision: {} recall: {}'.format(accuracy, precision, recall))

