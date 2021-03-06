from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.mutation import Mutation
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
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
    def __init__(self, model_f, ds, batch_size, img_size):
        self.model_f, self.ds, self.batch_size, self.img_size = model_f, ds, batch_size, img_size
        self.index, self.weights_shape = findShape(model_f(img_size).weights)  # find weights
        super(MyProblem, self).__init__(n_var=self.index[-1], n_obj=2, n_constr=0, xl=np.array([-1]*self.index[-1]),
                                        xu=np.array([1]*self.index[-1]))

    def _evaluate(self, x, out, *args, **kwargs):
        pop_num = x.shape[0]
        objs = np.zeros((pop_num, self.n_obj))
        # evaluate each individual
        model = self.model_f(self.img_size)
        for i in range(pop_num):
            model.set_weights(setShape(x[i], self.index, self.weights_shape))  # set weights
            outputs, targets = forward(model, self.ds, self.batch_size) # forward
            # fitness value
            precision, recall, accuracy, entropy = evalute(targets, outputs)
            accuracies.append(accuracy)
            entropys.append(cast(entropy, tf.float64).numpy())
            objs[i, 0] = 1 - precision
            objs[i, 1] = 1 - recall
        out["F"] = objs

class Initing(Sampling):
    def __init__(self, solution=[]):
        super(Initing, self).__init__()
        self.solution = solution

    def _do(self, problem, n_samples, **kwargs):
        global precisions, recalls, accuracies, entropys, hvs
        precisions = []
        recalls = []
        accuracies = []
        entropys = []
        hvs = []
        for _ in range(n_samples-len(self.solution)):
            w = np.random.normal(loc=0, scale=0.1, size=problem.n_var).astype(np.float32)
            self.solution.append(w)
        return np.array(self.solution)

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
          img_size=[28, 28], solution=[], modelName=None, dataName=None):

    datasets = {'mnist': mnsitDataset, 'cifar': cifarDataset, 'digits': digitsDataset}
    model = {'cnn': cnn, 'cnn2': cnn2, 'cnn3': cnn3, 'fnn2': fnn2, 'fnn3': fnn3}
    train_ds = datasets[dataName]
    myProblem = MyProblem(model[modelName], train_ds, batch_size, img_size)
    termination = get_termination("n_gen", n_gen)
    algorithm = NSGA2(pop_size=pop_size, sampling=Initing(solution))
    print('\nThe first step of optimization is in progress...')
    res = minimize(myProblem, algorithm, termination, seed=1, save_history=True, display=MyDisplay(), verbose=True)
    return res.X, res.pop.get("X"), res.F, precisions, recalls, accuracies, entropys, hvs


if __name__ == '__main__':
    X, pop, f = optim(10, 2, img_size=[32, 32], modelName='bp', dataName='cifar')
    print(f)