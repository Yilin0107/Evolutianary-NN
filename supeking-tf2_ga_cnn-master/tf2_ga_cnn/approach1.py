import genetic.GLMutation_NSGA2 as GLMutation_NSGA2
from common.utils import *
from pymoo.factory import get_performance_indicator
import warnings
import os
import sys
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


try:
    modelName, dataName= sys.argv[1:3]
except Exception as e:
    print(sys.argv)
    print(e)

solution = []
#modelName = 'fnn2'
#dataName = 'digits'
img_size = [28, 28, 1]
batch_size = 10000
if dataName == 'digits':
    img_size = [8, 8, 1]
elif dataName == 'mnist':
    img_size = [28, 28, 1]
elif dataName == 'cifar':
    img_size = [32, 32, 3]

n_ind = 100
n_gen = 10

eta1 = 4  # the number of groups for mutation
prob1 = None
pareto, pop, fit, precisions, recalls, accuracies, entropys, hvs = GLMutation_NSGA2.optim(n_ind, n_gen, batch_size,
                                                                                          eta1, prob1, img_size,
                                                                                          solution,
                                                                                          modelName,
                                                                                    dataName)

'''
# process data to be saved
pareto_recall= fit[:, 1]
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

'''

precisions, recalls, accuracies, entropys, final_accuracies, final_entropys = processData(n_gen, fit, precisions,
                                                                                          recalls, accuracies, entropys)

# save data
np.savez('./model/GLMutation_NSGA2/individuals_performance/{}_{}_Performance'.format(modelName, dataName),
         precision=precisions, recall=recalls, entropy=entropys, accuracy=accuracies)
np.savez('./model/GLMutation_NSGA2/hypervolumes/{}_{}_Hypervolume'.format(modelName, dataName), hypervolume=hvs)
np.savez('./model/GLMutation_NSGA2/final_performance/{}_{}_GLMutation_NSGA2'.format(modelName, dataName), pareto=pareto,
         precision=1 - fit[:, 0], recall=1 - fit[:, 1], entropy=final_entropys, accuracy=final_accuracies)
