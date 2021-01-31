import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, log_loss
import tensorflow as tf


def rgb2gray(rgb):
    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def evalute(target, output):
    output_ = np.argmax(output, axis=1)
    precision = precision_score(target, output_, average='macro')
    recall = recall_score(target, output_, average='macro')
    accuracy = accuracy_score(target, output_)
    target_onehot = tf.one_hot(target, depth=10)
    # entropy = log_loss(target_onehot, output)
    entropy = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(target, output))
    return precision, recall, accuracy, entropy


def forward(model, ds, batch_size, train=True):
    '''outputs, targets = [], []
    for image, _targets in ds(batch_size, train=train):
        _outputs = model(image)  # forward
        if len(outputs):
            outputs = np.concatenate([outputs, _outputs.numpy()], axis=0)
            targets = np.concatenate([targets, _targets], axis=0)
        else:
            outputs, targets = _outputs.numpy(), _targets'''
    image, targets = next(ds(batch_size, train=train))
    outputs = model(image)
    return outputs, targets


def findShape(weights):
    weights_shape = []
    index = [0]
    for i, w in enumerate(weights):
        weights_shape.append(w.shape.as_list())
        index.append(tf.size(w).numpy() + index[i])
    return index, weights_shape


def setShape(sample, index, weights_shape):
    weights = []
    for s, shape in enumerate(weights_shape):
        weights.append(sample[index[s]:index[s + 1]].reshape(shape))
    return weights


'''def GroupSeparation(X, n=4):
    index = np.argsort(np.abs(X))  # sort the indices of X
    individual_length = X.shape[1]
    split_index = individual_length % n

    index = np.array(np.split(index, n, axis=1))
    # ---- the indices are now split into n groups
    rand = np.random.randint(0, n)
    mutation_group = index[rand]
    mutation = np.full(X.shape, False)
    for step, i in enumerate(mutation):
        i[mutation_group[step]] = True
    return mutation'''


def GroupSeparation(X, n=4):
    index = np.argsort(np.abs(X))  # sort the indices of X
    individual_length = X.shape[1]
    left = int(individual_length % n)
    del_index = np.random.randint(individual_length - left, size=left)
    index = np.delete(index, del_index, axis=1)
    index = np.array(np.split(index, n, axis=1))
    # ---- the indices are now split into n groups
    rand = np.random.randint(0, n)
    mutation_group = index[rand]
    mutation = np.full(X.shape, False)
    for step, i in enumerate(mutation):
        i[mutation_group[step]] = True
    return mutation


def processData(n_gen, fit, precisions, recalls, accuracies, entropys):
    pareto_recall = fit[:, 1]
    pareto_index = []
    final_entropys = []
    final_accuracies = []

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    accuracies = np.array(accuracies)
    entropys = np.array(entropys)

    # precisions = np.array(np.split(precisions, n_gen))
    # recalls = np.array(np.split(recalls, n_gen))
    accuracies = np.array(np.split(accuracies, n_gen))
    entropys = np.array(np.split(entropys, n_gen))

    for r in pareto_recall:
        ind = np.where((recalls[n_gen - 1]) == r)[0][0]
        pareto_index.append(ind)

    for i in pareto_index:
        final_entropys.append(entropys[n_gen - 1][i])
        final_accuracies.append(accuracies[n_gen - 1][i])

    return precisions, recalls, accuracies, entropys, final_accuracies, final_entropys
