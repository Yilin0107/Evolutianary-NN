from common.load_data import mnsitDataset, cifarDataset
from common.utils import *
from common.net import cnn


model = cnn([28, 28])
dataset = mnsitDataset
weights_path = 'model/GLMutation_NSGA2/cnn_mnist_GLMutation_NSGA2.npz'
# load weights
index, weights_shape = findShape(model.weights)
#weights = np.load(weights_path)['pareto'][-1, ...]
weights = np.load(weights_path)['pareto']
for i in range(len(weights)):
    model.set_weights(setShape(weights[i, ...], index, weights_shape))
    outputs, targets = forward(model, dataset, batch_size=10000, train=False)
    precision, recall = evalute(targets, outputs)
    accuracy = accuracy_score(targets, np.argmax(outputs, axis=1))
    print('accuracy: {} precision: {} recall: {}'.format(accuracy, precision, recall)) # ALL FOUR INDICATORS