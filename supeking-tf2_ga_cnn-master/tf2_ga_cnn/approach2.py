import genetic.NSGA2 as NSGA2
import genetic.LinearCombination_NSGA2 as LinearCombination_NSGA2
import numpy as np
from common.utils import *
import warnings

warnings.filterwarnings("ignore")  # "error", "ignore", "always", "default", "module" or "once"

solution = []
modelName = 'fnn2'
dataName = 'digits'
batch_size = 10000
img_size = [8, 8, 1]

n_var1 = 40
n_var2 = 40
n_gen1 = 10
n_gen2 = 10  # n_eval


for ep in range(2):
    print('\nEpoch: {:02}'.format(ep))
    # standard NSGA2
    pareto, pop, fit, precisions, recalls, accuracies, entropys, hvs = NSGA2.optim(n_var1, n_gen1, batch_size, img_size, solution, modelName,
                                                                                   dataName)

    print(np.array(accuracies).shape)
    precisions, recalls, accuracies, entropys, final_accuracies, final_entropys = processData(n_gen1, fit, precisions,
                                                                                              recalls, accuracies,
                                                                                              entropys)
    np.savez('./model/LinearCombination_NSGA2/step1/individuals_performance/{}_{}-{:02}_Performance'.format(modelName, dataName, ep),
             precision=precisions, recall=recalls, entropy=entropys, accuracy=accuracies)

    np.savez('./model/LinearCombination_NSGA2/step1/hypervolumes/{}_{}-{:02}_Hypervolume'.format(modelName, dataName, ep), hypervolume=hvs)
    np.savez('./model/LinearCombination_NSGA2/step1/final_performance/{}_{}-{:02}_LinearCombination_NSGA2'.format(modelName, dataName, ep),
             pareto=pareto,
             precision=1 - fit[:, 0], recall=1 - fit[:, 1], entropy=final_entropys, accuracy=final_accuracies)


    if pareto.shape[0] == 1:
        print("currently, there is only one pareto solution")
        solution = pop
        continue
    # linear combination
    X, _, fit, precisions, recalls, accuracies, entropys, hvs = LinearCombination_NSGA2.optim(n_var2, n_gen2, batch_size, img_size, pareto, modelName,
                                              dataName)
    outputs = np.dot(X, pareto)
    for i in range(len(X)):
        pop = np.delete(pop, n_var1-i-1, axis=0)
    solution = np.concatenate([pop, outputs], axis=0)
    print(np.array(accuracies).shape)
    precisions, recalls, accuracies, entropys, final_accuracies, final_entropys = processData(n_gen2, fit, precisions,
                                                                                              recalls, accuracies,
                                                                                              entropys)
    np.savez('./model/LinearCombination_NSGA2/step2/individuals_performance/{}_{}-{:02}_Performance'.format(modelName,
                                                                                                            dataName,
                                                                                                            ep),
             precision=precisions, recall=recalls, entropy=entropys, accuracy=accuracies)

    np.savez(
        './model/LinearCombination_NSGA2/step2/hypervolumes/{}_{}-{:02}_Hypervolume'.format(modelName, dataName, ep),
        hypervolume=hvs)
    np.savez(
        './model/LinearCombination_NSGA2/step2/final_performance/{}_{}-{:02}_LinearCombination_NSGA2'.format(modelName,
                                                                                                             dataName,
                                                                                                             ep),
        pareto=pareto,
        precision=1 - fit[:, 0], recall=1 - fit[:, 1], entropy=final_entropys, accuracy=final_accuracies)


