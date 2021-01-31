import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_termination
from pymoo.operators.crossover.point_crossover import PointCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.mutation import Mutation
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
from pymoo.optimize import minimize

mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
data = tf.data.Dataset.from_tensor_slices((train_x.reshape([-1, 784])/255.0, train_y))
data = data.shuffle(buffer_size=60000).batch(10000)
test_data = tf.data.Dataset.from_tensor_slices((test_x.reshape([-1, 784])/255.0, test_y)).batch(10000)


def NN_evaluate_trainset(data, W1, b1):
    train_batch = next(iter(data))
    logits = tf.matmul(train_batch[0], W1) + b1
    train_preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(train_preds, train_batch[1]), tf.float32))
    xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=train_batch[1]))
    return acc, xent


class MyProblem(Problem):
    def __init__(self, n_var=7850, n_obj=2, n_constr=0, lb=-10, ub=10):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=lb, xu=ub)

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        for i in range(x.shape[0]):
            individual = x[i]
            # process the data
            W1 = tf.Variable(individual[:7840].reshape([784, 10]).astype(np.float32))
            b1 = tf.Variable(individual[7840:].astype(np.float32))
            # get performance and crossentropy
            # acc = NN_evaluate_testset(test_data,W1,b1)
            acc, xent = NN_evaluate_trainset(data, W1, b1)
            objs[i, 0] = 1 - acc
            objs[i, 1] = xent
        out["F"] = objs


# sampling: the initial population is sampled form the noral distribution
# Minist: 784*10+10
class MySampling(Sampling):

    # dims: the number of parameters of the network,
    def __init__(self):
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        val = np.random.normal(loc=0, scale=0.1, size=[n_samples, problem.n_var])
        return val


# n: number of groups
# n: var_n must be divisible by n(np.split)
def GroupSeparation(X, n=10):
    index = np.argsort(X)  # sort the indices of X
    index = np.array(np.split(index, n, axis=1))
    # ---- the indices are now splited into n groups
    rand = np.random.randint(0, n)
    mutation_group = index[rand]
    mutation = np.full(X.shape, False)
    for step, i in enumerate(mutation):
        i[mutation_group[step]] = True
    return mutation


class GLMutation(Mutation):
    def __init__(self, eta, prob=None):
        super().__init__()
        self.eta = float(eta)

        if prob is not None:
            self.prob = float(prob)
        else:
            self.prob = None

    def _do(self, problem, X, **kwargs):

        X = X.astype(np.float)
        do_mutation = GroupSeparation(X, 10)
        Y = np.full(X.shape, np.inf)
        # do_mutation has the same shape with X， and each 'True' elements are
        # the elements to be mutated

        Y[:, :] = X

        xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)[do_mutation]
        xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)[do_mutation]

        X = X[do_mutation]

        delta1 = (X - xl) / (xu - xl)
        delta2 = (xu - X) / (xu - xl)

        mut_pow = 1.0 / (self.eta + 1.0)

        #
        rand = np.full(X.shape, np.random.rand())
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        deltaq = np.zeros(X.shape)

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]

        # mutated values
        _Y = X + deltaq * (xu - xl)

        # back in bounds if necessary (floating point issues)
        _Y[_Y < xl] = xl[_Y < xl]
        _Y[_Y > xu] = xu[_Y > xu]

        # set the values for output
        Y[do_mutation] = _Y

        # in case out of bounds repair (very unlikely)
        Y = set_to_bounds_if_outside_by_problem(problem, Y)
        return Y


problem = MyProblem()

algorithm = NSGA2(
    pop_size=100,
    n_offsprings=100,
    sampling=MySampling(),  # get_sampling("real_random")
    # crossover = PointCrossover(n_points=10),
    mutation=GLMutation(eta=10, prob=0.6),
    eliminate_duplicates=True)

termination = get_termination("n_gen", 100)

res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=True)

print(res.pop.get("X"))
print(res.pop.get("F"))
