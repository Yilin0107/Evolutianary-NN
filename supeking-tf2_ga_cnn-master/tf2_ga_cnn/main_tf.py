from common.net import cnn, bp
from common.load_data import mnist, cifar
from common.utils import *


def training(modelName, dataName, img_size=[28, 28], epoch=25):
    datasets = {'mnist': mnist, 'cifar': cifar}
    _model = {'cnn': cnn, 'bp': bp}
    model = _model[modelName](img_size)
    (train_x, train_y), (test_x, test_y) = datasets[dataName].load_data()
    #if train_x.shape[-1]==3:
        #train_x, test_x = rgb2gray(train_x), rgb2gray(test_x)
    train_x, test_x = (train_x / 255.0).astype(np.float32), (test_x / 255.0).astype(np.float32)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # FOUR INDICATORS
    model.fit(train_x, train_y, epochs=epoch, validation_data=(test_x, test_y))
    model.save('model/{}_{}.h5'.format(modelName, dataName))
    outputs = model(test_x)
    precision, recall = evalute(test_y, outputs.numpy())
    accuracy = accuracy_score(test_y, np.argmax(outputs, axis=1))
    print('accuracy: {} precision: {} recall: {}'.format(accuracy, precision, recall))

training('cnn', 'mnist', img_size=[28, 28], epoch=5)






