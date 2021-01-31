import genetic.GLMutation_NSGA2 as step1
import genetic.LinearCombination_NSGA2 as step2
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # "error", "ignore", "always", "default", "module" or "once"

solution = []
modelName = 'bp'
#dataName = 'cifar'
dataName = 'mnist'
batch_size = 10000
img_size = [28, 28]

n_var1 = 100
n_var2 = 100
n_gen1 = 40
n_gen2 = 40

eta1 = 10
eta2 = None
prob1 = None
prob2 = None

for ep in range(1):
    print('\nEpoch: {:02}'.format(ep))
    # 第一步优化
    pareto, pop, fit = step1.optim(n_var1, n_gen1, batch_size, eta1, prob1, img_size, solution, modelName, dataName)
    print('step1_fit: {}'.format(1 - fit))
    np.savez('./model/{}_{}-{:02}_step1'.format(modelName, dataName, ep), pareto=pareto, fit=fit)

    # 第二步优化
    X, _, fit = step2.optim(n_var2, n_gen2, batch_size, eta2, prob2, img_size, pareto, modelName, dataName)
    print('step2_fit: {}'.format(1 - fit))
    outputs = np.dot(X, pareto)
    solution = np.concatenate([pop, outputs], axis=0)
    np.savez('./model/{}_{}-{:02}_step2'.format(modelName, dataName, ep), pareto=outputs, fit=fit)



