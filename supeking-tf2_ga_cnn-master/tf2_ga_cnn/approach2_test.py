from common.load_data import mnsitDataset, cifarDataset
from common.utils import *
from common.net import bp


model = bp([28, 28])
dataset = mnsitDataset
weights_path = 'model/LinearCombination_NSGA2/bp_mnist-00_LinearCombination_NSGA2.npz'
# load weights
index, weights_shape = findShape(model.weights)
weights = np.load(weights_path)['pareto']
for i in range(len(weights)):
    model.set_weights(setShape(weights[i, ...], index, weights_shape))
    outputs, targets = forward(model, dataset, batch_size=10000)
    precision, recall = evalute(targets, outputs)
    accuracy = accuracy_score(targets, np.argmax(outputs, axis=1))
    print('accuracy: {} precision: {} recall: {}'.format(accuracy, precision, recall))