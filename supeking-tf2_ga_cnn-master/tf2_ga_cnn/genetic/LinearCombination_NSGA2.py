from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation

from genetic.GLMutation_NSGA2 import MyProblem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.mutation import Mutation
from pymoo.optimize import minimize
from pymoo.util.display import Display
from common.net import cnn, fnn2, fnn3, cnn2, cnn3
from common.load_data import mnsitDataset, cifarDataset, digitsDataset
from pymoo.factory import get_performance_indicator
from common.utils import *
from tensorflow import cast


precisions = []
recalls = []
accuracies = []
entropys = []
hvs = []


class MyProblem(Problem):
    def __init__(self, pareto, model_f, ds, batch_size, img_size):
        self.model_f, self.ds, self.batch_size, self.img_size = model_f, ds, batch_size, img_size
        self.index, self.weights_shape = findShape(model_f(img_size).weights)  # find weights
        self.pareto = pareto
        super(MyProblem, self).__init__(n_var=len(pareto), n_obj=2, n_constr=0, xl=np.array([-1]*len(pareto)),
                                        xu=np.array([1]*len(pareto)))

    def _evaluate(self, x, out, *args, **kwargs):
        pop_num = x.shape[0]
        objs = np.zeros((pop_num, self.n_obj))
        # 计算每个个体的适应度
        model = self.model_f(self.img_size)
        for i in range(pop_num):
            _x = x[i].reshape((1, -1)).dot(self.pareto).reshape(-1)
            model.set_weights(setShape(_x, self.index, self.weights_shape))  # set weights
            outputs, targets = forward(model, self.ds, self.batch_size) # forward
            # 计算相关指标
            precision, recall, accuracy, entropy = evalute(targets, outputs)
            accuracies.append(accuracy)
            entropys.append(cast(entropy, tf.float64).numpy())
            objs[i, 0] = 1 - precision
            objs[i, 1] = 1 - recall
        out["F"] = objs


class GLMutation(Mutation):
    def __init__(self, eta, prob=None):
        super(GLMutation, self).__init__()
        self.prob = prob  # 每个样本变异的概率
        self.eta = eta  # 样本中基因变异的概率

    def _do(self, problem, X, **kwargs):
        for i, x in enumerate(X):
            if np.random.rand() <= self.prob:
                ind = np.random.rand(x.shape[0]) <= self.eta
                X[i][ind] = -1 + 2*np.random.random(ind.sum())
        return X


class Initing(Sampling):
    def __init__(self):
        super(Initing, self).__init__()

    def _do(self, problem, n_samples, **kwargs):
        global precisions, recalls, accuracies, entropys, hvs
        precisions = []
        recalls = []
        accuracies = []
        entropys = []
        hvs = []
        solution = []
        for _ in range(n_samples):
            solution.append(-1 + 2*np.random.random((1, problem.n_var)))
        return np.concatenate(solution, axis=0)

class MyDisplay(Display):
    def _do(self, problem, evaluator, algorithm):
        super(MyDisplay, self)._do(problem,evaluator,algorithm)
        self.output.append("Obj1_avg",np.mean(algorithm.pop.get("F")[:, 0]))
        self.output.append("Obj2_avg", np.mean(algorithm.pop.get("F")[:, 1]))
        hv = get_performance_indicator("hv", ref_point=np.array([1, 1]))
        hv_value = hv.calc(algorithm.pop.get("F"))
        self.output.append("hv", hv_value)
        hvs.append(hv_value)
        precisions.append(algorithm.pop.get("F")[:, 0])
        recalls.append(algorithm.pop.get("F")[:, 1])


def optim(pop_size=10, n_gen=10, batch_size=10000,
          img_size=[28, 28], pareto=None, modelName=None, dataName=None):

    datasets = {'mnist': mnsitDataset, 'cifar': cifarDataset, 'digits': digitsDataset}
    model = {'cnn': cnn, 'cnn2': cnn2, 'cnn3': cnn3, 'fnn2': fnn2, 'fnn3': fnn3}
    train_ds = datasets[dataName]
    myProblem = MyProblem(pareto, model[modelName], train_ds, batch_size, img_size)
    termination = get_termination("n_gen", n_gen)
    #termination = get_termination("n_eval", 1000)
    #algorithm = NSGA2(pop_size=pop_size, sampling=Initing(), mutation=GLMutation(eta=eta, prob=prob))
    algorithm = NSGA2(pop_size=pop_size, sampling=Initing())
    print('\nThe second step of optimization is in progress...')
    res = minimize(myProblem, algorithm, termination, seed=1, save_history=True, display=MyDisplay(), verbose=True)
    return res.X, res.pop.get("X"), res.F, precisions, recalls, accuracies, entropys, hvs


if __name__ == "__main__":
    pareto = np.random.rand((5*7850)).reshape(5, 7850)
    X, pop, f = optim(10, 2, pareto=pareto, img_size=[28, 28], modelName='bp', dataName='mnist')
    print(f)