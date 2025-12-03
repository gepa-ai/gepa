"""
This file offers access to functions used during the development of the article
    A Stratified Analysis of Bayesian Optimization Methods

It incorporates functions developed/collected for the AMPGO benchmark by Andrea Gavana <andrea.gavana@gmail.com>
As of January 2016, the website http://infinity77.net/global_optimization/test_functions.html hosts images
of some of these functions.

Each function has an evaluate, which accepts in a single axis numpy.array x and returns a single value.
None of these functions allows for vectorized computation.

NOTE: These functions are designed to be minimized ... the paper referenced above talks about maximization.
    This was intentional to fit the standard of most optimization algorithms.

Some of these functions work in different dimensions, and some have a specified dimension.  The assert statement
will prevent incorrect calls.

For some of these functions, the maximum and minimum are determined analytically.  For others, there is only
a numerical determination of the optimum.  Also, some of these functions have the same minimum value at multiple
locations; if that is the case, only the location of one is provided.

Each function is also tagged with a list of relevant classifiers:
  boring - A mostly boring function that only has a small region of action.
  oscillatory - A function with a general trend and an short range oscillatory component.
  discrete - A function which can only take discrete values.
  unimodal - A function with a single local minimum, or no local minimum and only a minimum on the boundary.

  multimodal - A function with multiple local minimum
  bound_min - A function with its minimum on the boundary.
  multi_min - A function which takes its minimum value at multiple locations.
  nonsmooth - A function with discontinuous derivatives.
  noisy - A function with a base behavior which is clouded by noise.
  unscaled - A function with max value on a grossly different scale than the average or min value.
  complicated - These are functions that may fit a behavior, but not in the most obvious or satisfying way.

The complicated tag is used to alert users that the function may have interesting or complicated behavior.
As an example, the Ackley function is technically oscillatory, but with such a short wavelength that its
behavior probably seems more like noise.  Additionally, it is meant to act as unimodal, but is definitely
not, and it may appear mostly boring in higher dimensions.
"""

from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy

from numpy import abs, arange, arctan2, asarray, cos, exp, floor, log, log10, mean
from numpy import pi, prod, roll, seterr, sign, sin, sqrt, sum, zeros, zeros_like, tan
from numpy import dot, inner

from scipy.special import jv as besselj

from timeit import default_timer as now

seterr(all="ignore")


def lzip(*args):
    """
    Zip, but returns zipped result as a list.
    """
    return list(zip(*args))


def execute_random_search(num_fevals, num_trials, function):
    """
    This function shows how to use this library in a sequential optimization setting.

    :param num_fevals: number of function evaluations available for the optimization
    :type num_fevals: int
    :param num_trials: number of trials to conduct; needed to understand randomness of the optimization
    :type num_trials: int
    :param function: The function object whose properties we want to test
    :type function: TestFunction
    :return: the fbest history of all of the optimization trials
    :rtype: numpy.array of size (num_trials, num_fevals)
    """

    # This function could also take in any other parameters required for the next points determination
    # For instance, most sequential optimizers use the previous observations to get better results
    def random_search_next_point(bounds):
        np_bounds = asarray(list(bounds))
        return np_bounds[:, 0] + (
            np_bounds[:, 1] - np_bounds[:, 0]
        ) * numpy.random.random(len(np_bounds))

    f_best_hist = numpy.empty((num_trials, num_fevals))
    for this_trial in range(num_trials):
        for this_feval in range(num_fevals):
            next_point = random_search_next_point(function.bounds)
            f_current = function.evaluate(next_point)
            if this_feval == 0:
                f_best_hist[this_trial, 0] = f_current
            else:
                f_best_hist[this_trial, this_feval] = min(
                    f_current, f_best_hist[this_trial, this_feval - 1]
                )

    return f_best_hist


class TestFunction(object):
    """
    The base class from which functions of interest inherit.
    """

    __metaclass__ = ABCMeta

    def __init__(self, dim, verify=True):
        assert dim > 0
        self.dim = dim
        self.verify = verify
        self.num_evals = 0
        self.min_loc = None
        self.fmin = None
        self.local_fmin = []
        self.fmax = None
        self.bounds = None
        self.classifiers = []

        # Note(Mike) - Not using the records yet, but will be soon
        self.records = None
        self.reset_records()

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.dim)

    def evaluate(self, x):
        if self.verify and (not isinstance(x, numpy.ndarray) or x.shape != (self.dim,)):
            raise ValueError(
                "Argument must be a numpy array of length {}".format(self.dim)
            )

        self.num_evals += 1
        value = self.do_evaluate(x)
        to_be_returned = value.item() if hasattr(value, "item") else value
        self.update_records(now(), x, to_be_returned)
        # Convert numpy types to Python types
        return to_be_returned

    def update_records(self, time, location, value):
        self.records["time"].append(time)
        self.records["locations"].append(location)
        self.records["values"].append(value)

    def reset_records(self):
        self.records = {"time": [], "locations": [], "values": []}

    @abstractmethod
    def do_evaluate(self, x):
        """
        :param x: point at which to evaluate the function
        :type x: numpy.array with shape (self.dim, )
        """
        raise NotImplementedError


class Discretizer(TestFunction):
    """
    This class converts function evaluations into discrete values at a desired resolution.

    If res == 4, the interval [0,1) will have 4 distinct values: {0, 0.25, 0.5, 0.75}.
    If res == .25, the interval [0,10) will have 3 distinct values: {0, 4, 8}.

    Example: ackley_res1 = Discretizer(Ackley(), 1)
    """

    def __init__(self, func, res, verify=True):
        assert isinstance(func, TestFunction)
        if res <= 0:
            raise ValueError("Resolution level must be positive, level={0}".format(res))
        super(Discretizer, self).__init__(func.dim, verify)
        self.bounds, self.min_loc = func.bounds, func.min_loc
        self.res = res
        self.fmax = numpy.floor(self.res * func.fmax) / self.res
        self.fmin = numpy.floor(self.res * func.fmin) / self.res
        self.func = func
        self.classifiers = list(set(self.classifiers) | set(["discrete"]))

    def do_evaluate(self, x):
        return numpy.floor(self.res * self.func.evaluate(x)) / self.res

    def __repr__(self):
        return "{0}({1!r}, {2})".format(
            self.__class__.__name__,
            self.func,
            self.res,
        )


class Failifier(TestFunction):
    """
    This class renders certain parts of the domain into failure regions, and returns a 'nan' at those points.

    You must define a function that can return whether or not a certain point is in a failure region.
    Instead of returning a 'nan', this can also be set up to return the worst value possible, which is useful
    for comparison against methods that cannot manage failure cases.

    Some common functions are provided within this class as static methods.

    Example:    failure_function = lambda x: Failifier.in_n_sphere(x, numpy.zeros_like(x), 1, 5)
                alpine01_fail = Failifier(Alpine01(), failure_function)
            This would produce failures outside of the ring between the origin circles of radius 1 and radius 5
    """

    @staticmethod
    def in_2d_rectangle(x, x1_1, x1_2, x2_1, x2_2):
        return x1_1 <= x[0] <= x1_2 and x2_1 <= x[1] <= x2_2

    @staticmethod
    def in_n_sphere(x, c, r1, r2):
        radius = sqrt(sum([(a - b) ** 2 for a, b in zip(x, c)]))
        return r1 <= radius <= r2

    @staticmethod
    def sum_to_lte(x, metric):
        return sum(x) <= metric

    @staticmethod
    def linear_constraint(x, weights, metric):
        return inner(x, weights) <= metric

    @staticmethod
    def each_lte(x, metric):
        for _x in x:
            if _x > metric:
                return False
        return True

    # Note(Mike) - This is not really a simplex, more like the 1-norm.  But it's fine
    @staticmethod
    def in_simplex(x, c):
        if sum(abs(x)) <= c:
            return True
        return False

    @staticmethod
    def at_midpoint(x, bounds):
        if all(
            this_dim_x == 0.5 * sum(this_dim_bound)
            for this_dim_x, this_dim_bound in zip(x, bounds)
        ):
            return True
        return False

    def __init__(self, func, fail_indicator, return_nan=True, verify=True):
        assert isinstance(func, TestFunction)
        super(Failifier, self).__init__(func.dim, verify)
        self.bounds, self.min_loc, self.fmax, self.fmin = (
            func.bounds,
            func.min_loc,
            func.fmax,
            func.fmin,
        )
        self.func = func
        self.fail_indicator = fail_indicator
        self.return_nan = return_nan
        self.classifiers = list(set(self.classifiers) | set(["failure"]))

    def do_evaluate(self, x):
        if self.fail_indicator(x):
            if self.return_nan:
                return float("nan")
            else:
                return self.fmax
        else:
            return self.func.evaluate(x)

    def __repr__(self):
        return "{0}({1!r}, failure)".format(
            self.__class__.__name__,
            self.func,
        )


class Constrainer(TestFunction):
    """This class defines a set of (linear) constraints to the imput space.

    The constraints should be defined as a matrix (list of lists) of
    weights "A" and the list of right hand side (rhs) terms "b" such that the
    parameters x should satisfy A*x => b

    We have the failify boolean flag (True by default), to consider
    the constrained space as failure region to help comparison with
    methods that do not support hard constraints.

    Example:    weights = [[1,  1]
                           [1, -1]]
                rhs = [1, 2.5]
                alpine01_constrained = Constrainer(Alpine01(), weights, rhs, failify=True)
            This would generate the constraints that, if the input space is (x,y), then x + y => 1 and x - y => 2.5

    """

    @staticmethod
    def default_constraint_check(x, weights, rhs):
        for w, r in zip(weights, rhs):
            if inner(x, w) < r:
                return False
        return True

    def __init__(
        self,
        func,
        constraint_weights,
        constraint_rhs,
        constraint_check=None,
        return_nan=True,
        verify=True,
    ):
        assert isinstance(func, TestFunction)
        assert len(constraint_weights) == len(constraint_rhs)
        super(Constrainer, self).__init__(func.dim, verify)
        self.bounds, self.min_loc, self.fmax, self.fmin = (
            func.bounds,
            func.min_loc,
            func.fmax,
            func.fmin,
        )
        self.func = func
        self.constraint_weights = constraint_weights
        self.constraint_rhs = constraint_rhs
        self.return_nan = return_nan
        self.classifiers = list(set(self.classifiers) | set(["constraint"]))
        if constraint_check is not None:
            self.constraint_check = constraint_check
        else:
            self.constraint_check = Constrainer.default_constraint_check

    def do_evaluate(self, x):
        if self.constraint_check is not None and self.constraint_check(
            x, self.constraint_weights, self.constraint_rhs
        ):
            return self.func.evaluate(x)
        elif self.return_nan:
            return float("nan")
        else:
            return self.fmax

    def __repr__(self):
        return "{0}({1!r}, constraint)".format(
            self.__class__.__name__,
            self.func,
        )


class Noisifier(TestFunction):
    """
    This class dirties function evaluations with Gaussian noise.

    If type == 'add', then the noise is additive; for type == 'mult' the noise is multiplicative.
    sd defines the magnitude of the noise, i.e., the standard deviation of the Gaussian.

    Example: ackley_noise_addp01 = Noisifier(Ackley(3), 'add', .01)

    Obviously, with the presence of noise, the max and min may no longer be accurate.
    """

    def __init__(self, func, noise_type, level, verify=True):
        assert isinstance(func, TestFunction)
        if level <= 0:
            raise ValueError("Noise level must be positive, level={0}".format(level))
        super(Noisifier, self).__init__(func.dim, verify)
        self.bounds, self.min_loc, self.fmax, self.fmin = (
            func.bounds,
            func.min_loc,
            func.fmax,
            func.fmin,
        )
        self.type = noise_type
        self.level = level
        self.func = func
        self.classifiers = list(set(self.classifiers) | set(["noisy"]))

    def do_evaluate(self, x):
        if self.type == "add":
            return self.func.evaluate(x) + self.level * numpy.random.normal()
        else:
            return self.func.evaluate(x) * (1 + self.level * numpy.random.normal())

    def __repr__(self):
        return "{0}({1!r}, {2}, {3})".format(
            self.__class__.__name__,
            self.func,
            self.type,
            self.level,
        )


class Ackley(TestFunction):
    def __init__(self, dim=2):
        super(Ackley, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [30] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 22.26946404462
        self.classifiers = ["complicated", "oscillatory", "unimodal", "noisy"]

    def do_evaluate(self, x):
        a = 20
        b = 0.2
        c = 2 * pi
        return (
            -a * exp(-b * sqrt(1.0 / self.dim * sum(x**2)))
            - exp(1.0 / self.dim * sum(cos(c * x)))
            + a
            + exp(1)
        )


class Adjiman(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Adjiman, self).__init__(dim)
        self.bounds = ([-1, 2], [-1, 1])
        self.min_loc = [2, 0.10578]
        self.fmin = -2.02180678
        self.fmax = 1.07715029333
        self.classifiers = ["unimodal", "bound_min"]

    def do_evaluate(self, x):
        x1, x2 = x
        return cos(x1) * sin(x2) - x1 / (x2**2 + 1)


class Alpine01(TestFunction):
    def __init__(self, dim=2):
        super(Alpine01, self).__init__(dim)
        self.bounds = lzip([-6] * self.dim, [10] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 8.71520568065 * self.dim
        self.classifiers = ["nonsmooth"]

    def do_evaluate(self, x):
        return sum(abs(x * sin(x) + 0.1 * x))


class Alpine02(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Alpine02, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [10] * self.dim)
        self.min_loc = [7.91705268, 4.81584232]
        self.fmin = -6.12950389113
        self.fmax = 7.88560072413
        self.classifiers = ["oscillatory", "multi_min"]

    def do_evaluate(self, x):
        return prod(sqrt(x) * sin(x))


class ArithmeticGeometricMean(TestFunction):
    def __init__(self, dim=2):
        super(ArithmeticGeometricMean, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [10] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = (10 * (self.dim - 1.0) / self.dim) ** 2
        self.classifiers = ["bound_min", "boring", "multi_min"]

    def do_evaluate(self, x):
        return (mean(x) - prod(x) ** (1.0 / self.dim)) ** 2


class BartelsConn(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(BartelsConn, self).__init__(dim)
        self.bounds = lzip([-2] * self.dim, [5] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 1
        self.fmax = 76.2425864601
        self.classifiers = ["nonsmooth", "unimodal"]

    def do_evaluate(self, x):
        x1, x2 = x
        return abs(x1**2 + x2**2 + x1 * x2) + abs(sin(x1)) + abs(cos(x2))


class Beale(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Beale, self).__init__(dim)
        self.bounds = lzip([-4.5] * self.dim, [4.5] * self.dim)
        self.min_loc = [3, 0.5]
        self.fmin = 0
        self.fmax = 181853.613281
        self.classifiers = ["boring", "unscaled"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            (1.5 - x1 + x1 * x2) ** 2
            + (2.25 - x1 + x1 * x2**2) ** 2
            + (2.625 - x1 + x1 * x2**3) ** 2
        )


class Bird(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Bird, self).__init__(dim)
        self.bounds = lzip([-2 * pi] * self.dim, [2 * pi] * self.dim)
        self.min_loc = [4.701055751981055, 3.152946019601391]
        self.fmin = -64.60664462282
        self.fmax = 160.63195224589
        self.classifiers = ["multi_min"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            sin(x1) * exp((1 - cos(x2)) ** 2)
            + cos(x1) * exp((1 - sin(x1)) ** 2)
            + (x1 - x2) ** 2
        )


class Bohachevsky(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Bohachevsky, self).__init__(dim)
        self.bounds = lzip([-15] * self.dim, [8] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 675.6
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        x1, x2 = x
        return x1**2 + 2 * x2**2 - 0.3 * cos(3 * pi * x1) - 0.4 * cos(4 * pi * x2) + 0.7


class BoxBetts(TestFunction):
    def __init__(self, dim=3):
        assert dim == 3
        super(BoxBetts, self).__init__(dim)
        self.bounds = ([0.9, 1.2], [9, 11.2], [0.9, 1.2])
        self.min_loc = [1, 10, 1]
        self.fmin = 0
        self.fmax = 0.28964792415
        self.classifiers = ["boring"]

    def do_evaluate(self, x):
        x1, x2, x3 = x
        return sum(
            [
                (
                    exp(-0.1 * i * x1)
                    - exp(-0.1 * i * x2)
                    - (exp(-0.1 * i) - exp(-i)) * x3
                )
                ** 2
                for i in range(2, 12)
            ]
        )


class Branin01(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Branin01, self).__init__(dim)
        self.bounds = [[-5, 10], [0, 15]]
        self.min_loc = [-pi, 12.275]
        self.fmin = 0.39788735772973816
        self.fmax = 308.129096012
        self.classifiers = ["multi_min"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            (x2 - (5.1 / (4 * pi**2)) * x1**2 + 5 * x1 / pi - 6) ** 2
            + 10 * (1 - 1 / (8 * pi)) * cos(x1)
            + 10
        )


class Branin02(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Branin02, self).__init__(dim)
        self.bounds = [(-5, 15), (-5, 15)]
        self.min_loc = [-3.2, 12.53]
        self.fmin = 5.559037
        self.fmax = 506.983390872

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            (x2 - (5.1 / (4 * pi**2)) * x1**2 + 5 * x1 / pi - 6) ** 2
            + 10 * (1 - 1 / (8 * pi)) * cos(x1) * cos(x2)
            + log(x1**2 + x2**2 + 1)
            + 10
        )


class Brent(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Brent, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [-10] * self.dim
        self.fmin = 0
        self.fmax = 800
        self.classifiers = ["unimodal", "bound_min"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (x1 + 10) ** 2 + (x2 + 10) ** 2 + exp(-(x1**2) - x2**2)


class Brown(TestFunction):
    def __init__(self, dim=2):
        assert dim > 1
        super(Brown, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [2] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(numpy.array([2] * self.dim))
        self.classifiers = ["unimodal", "unscaled"]

    def do_evaluate(self, x):
        x0 = x[:-1]
        x1 = x[1:]
        return sum((x0**2) ** (x1**2 + 1) + (x1**2) ** (x0**2 + 1))


class Bukin06(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Bukin06, self).__init__(dim)
        self.bounds = [(-15, -5), (-3, 3)]
        self.min_loc = [-10, 1]
        self.fmin = 0
        self.fmax = 229.178784748
        self.classifiers = ["nonsmooth"]

    def do_evaluate(self, x):
        x1, x2 = x
        return 100 * sqrt(abs(x2 - 0.01 * x1**2)) + 0.01 * abs(x1 + 10)


class CarromTable(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(CarromTable, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [9.646157266348881, 9.646134286497169]
        self.fmin = -24.15681551650653
        self.fmax = 0
        self.classifiers = ["boring", "multi_min", "nonsmooth", "complicated"]

    def do_evaluate(self, x):
        x1, x2 = x
        return -((cos(x1) * cos(x2) * exp(abs(1 - sqrt(x1**2 + x2**2) / pi))) ** 2) / 30


class Chichinadze(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Chichinadze, self).__init__(dim)
        self.bounds = lzip([-30] * self.dim, [30] * self.dim)
        self.min_loc = [6.189866586965680, 0.5]
        self.fmin = -42.94438701899098
        self.fmax = 1261
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            x1**2
            - 12 * x1
            + 11
            + 10 * cos(pi * x1 / 2)
            + 8 * sin(5 * pi * x1 / 2)
            - 0.2 * sqrt(5) * exp(-0.5 * ((x2 - 0.5) ** 2))
        )


class Cigar(TestFunction):
    def __init__(self, dim=2):
        assert dim > 1
        super(Cigar, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [1] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 1 + 1e6 * self.dim
        self.classifiers = ["unimodal", "unscaled"]

    def do_evaluate(self, x):
        return x[0] ** 2 + 1e6 * sum(x[1:] ** 2)


class Cola(TestFunction):
    def __init__(self, dim=17):
        assert dim == 17
        super(Cola, self).__init__(dim)
        self.bounds = [[0, 4]] + list(lzip([-4] * (self.dim - 1), [4] * (self.dim - 1)))
        self.min_loc = [
            0.651906,
            1.30194,
            0.099242,
            -0.883791,
            -0.8796,
            0.204651,
            -3.28414,
            0.851188,
            -3.46245,
            2.53245,
            -0.895246,
            1.40992,
            -3.07367,
            1.96257,
            -2.97872,
            -0.807849,
            -1.68978,
        ]
        self.fmin = 11.7464
        self.fmax = 1607.73849331

    def do_evaluate(self, x):
        d = asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1.27, 0, 0, 0, 0, 0, 0, 0, 0],
                [1.69, 1.43, 0, 0, 0, 0, 0, 0, 0],
                [2.04, 2.35, 2.43, 0, 0, 0, 0, 0, 0],
                [3.09, 3.18, 3.26, 2.85, 0, 0, 0, 0, 0],
                [3.20, 3.22, 3.27, 2.88, 1.55, 0, 0, 0, 0],
                [2.86, 2.56, 2.58, 2.59, 3.12, 3.06, 0, 0, 0],
                [3.17, 3.18, 3.18, 3.12, 1.31, 1.64, 3, 0, 0],
                [3.21, 3.18, 3.18, 3.17, 1.7, 1.36, 2.95, 1.32, 0],
                [2.38, 2.31, 2.42, 1.94, 2.85, 2.81, 2.56, 2.91, 2.97],
            ]
        )
        x1 = asarray([0, x[0]] + list(x[1::2]))
        x2 = asarray([0, 0] + list(x[2::2]))
        return sum(
            [
                sum(
                    (sqrt((x1[i] - x1[0:i]) ** 2 + (x2[i] - x2[0:i]) ** 2) - d[i, 0:i])
                    ** 2
                )
                for i in range(1, len(x1))
            ]
        )


class Corana(TestFunction):
    def __init__(self, dim=4):
        assert dim == 4
        super(Corana, self).__init__(dim)
        self.bounds = lzip([-5] * self.dim, [5] * self.dim)
        self.min_loc = [0] * self.dim
        self.fglob = 0
        self.fmin = 0
        self.fmax = 24999.3261012
        self.classifiers = ["boring", "unscaled", "nonsmooth"]

    def do_evaluate(self, x):
        d = [1, 1000, 10, 100]
        r = 0
        for j in range(4):
            zj = floor(abs(x[j] / 0.2) + 0.49999) * sign(x[j]) * 0.2
            if abs(x[j] - zj) < 0.05:
                r += 0.15 * ((zj - 0.05 * sign(zj)) ** 2) * d[j]
            else:
                r += d[j] * x[j] * x[j]
        return r


class CosineMixture(TestFunction):
    def __init__(self, dim=2):
        super(CosineMixture, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [1] * self.dim)
        self.min_loc = [0.184872823182918] * self.dim
        self.fmin = -0.063012202176250 * self.dim
        self.fmax = 0.9 * self.dim
        self.classifiers = ["oscillatory", "multi_min"]

    def do_evaluate(self, x):
        return 0.1 * sum(cos(5 * pi * x)) + sum(x**2)


class CrossInTray(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(CrossInTray, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [1.349406685353340, 1.349406608602084]
        self.fmin = -2.062611870822739
        self.fmax = -0.25801263059
        self.classifiers = ["oscillatory", "multi_min", "nonsmooth", "complicated"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            -0.0001
            * (abs(sin(x1) * sin(x2) * exp(abs(100 - sqrt(x1**2 + x2**2) / pi))) + 1)
            ** 0.1
        )


class Csendes(TestFunction):
    def __init__(self, dim=2):
        super(Csendes, self).__init__(dim)
        self.bounds = lzip([-0.5] * self.dim, [1] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(asarray([1] * self.dim))
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        return sum((x**6) * (2 + sin(1 / (x + numpy.finfo(float).eps))))


class Cube(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Cube, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [1] * self.dim
        self.fmin = 0
        self.fmax = 102010121
        self.classifiers = ["unimodal", "boring", "unscaled"]

    def do_evaluate(self, x):
        x1, x2 = x
        return 100 * (x2 - x1**3) ** 2 + (1 - x1) ** 2


class Damavandi(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Damavandi, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [14] * self.dim)
        self.min_loc = [2] * self.dim
        self.fmin = 0
        self.fmax = 149

    def do_evaluate(self, x):
        x1, x2 = x
        t1, t2 = pi * (x1 - 2), pi * (x2 - 2)
        if abs(x1 - 2) > 1e-3 and abs(x2 - 2) > 1e-3:
            numerator = sin(t1) * sin(t2)
            denominator = t1 * t2
            quotient = numerator / denominator
        else:
            x_term = 1 - t1**2 / 6 if abs(x1 - 2) <= 1e-3 else sin(t1) / t1
            y_term = 1 - t2**2 / 6 if abs(x2 - 2) <= 1e-3 else sin(t2) / t2
            quotient = x_term * y_term
        factor1 = 1 - (abs(quotient)) ** 5
        factor2 = 2 + (x1 - 7) ** 2 + 2 * (x2 - 7) ** 2
        return factor1 * factor2


class Deb01(TestFunction):
    def __init__(self, dim=2):
        super(Deb01, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [1] * self.dim)
        self.min_loc = [0.3] * self.dim
        self.fmin = -1
        self.fmax = 0
        self.classifiers = ["oscillatory", "multi_min"]

    def do_evaluate(self, x):
        return -(1.0 / self.dim) * sum(sin(5 * pi * x) ** 6)


class Deb02(TestFunction):
    def __init__(self, dim=2):
        super(Deb02, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.min_loc = [0.0796993926887] * self.dim
        self.fmin = -1
        self.fmax = 0
        self.classifiers = ["oscillatory", "multi_min"]

    def do_evaluate(self, x):
        return -(1.0 / self.dim) * sum(sin(5 * pi * (x**0.75 - 0.05)) ** 6)


class Deceptive(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Deceptive, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.min_loc = [0.333333, 0.6666666]
        self.fmin = -1
        self.fmax = 0
        self.classifiers = ["nonsmooth"]

    def do_evaluate(self, x):
        alpha = asarray(self.min_loc)
        beta = 2
        g = zeros((self.dim,))
        for i in range(self.dim):
            if x[i] <= 0:
                g[i] = x[i]
            elif x[i] < 0.8 * alpha[i]:
                g[i] = -x[i] / alpha[i] + 0.8
            elif x[i] < alpha[i]:
                g[i] = 5 * x[i] / alpha[i] - 4
            elif x[i] < (1 + 4 * alpha[i]) / 5:
                g[i] = 5 * (x[i] - alpha[i]) / (alpha[i] - 1) + 1
            elif x[i] <= 1:
                g[i] = (x[i] - 1) / (1 - alpha[i]) + 0.8
            else:
                g[i] = x[i] - 1
        return -(((1.0 / self.dim) * sum(g)) ** beta)


class DeflectedCorrugatedSpring(TestFunction):
    def __init__(self, dim=2):
        super(DeflectedCorrugatedSpring, self).__init__(dim)
        self.alpha = 5.0
        self.K = 5.0
        self.bounds = lzip([0] * self.dim, [1.5 * self.alpha] * self.dim)
        self.min_loc = [self.alpha] * self.dim
        self.fmin = self.do_evaluate(asarray(self.min_loc))
        self.fmax = self.do_evaluate(zeros(self.dim))
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        return -cos(self.K * sqrt(sum((x - self.alpha) ** 2))) + 0.1 * sum(
            (x - self.alpha) ** 2
        )


class Dolan(TestFunction):
    def __init__(self, dim=5):
        assert dim == 5
        super(Dolan, self).__init__(dim)
        self.bounds = lzip([-100] * self.dim, [20] * self.dim)
        self.min_loc = [94.3818, 43.4208, 44.8427, -40.2365, -21.0455]
        self.fmin = 0
        self.fmax = 2491.1585548
        self.classifiers = ["nonsmooth", "oscillatory", "multi_min"]

    def do_evaluate(self, x):
        x1, x2, x3, x4, x5 = x
        return abs(
            (x1 + 1.7 * x2) * sin(x1)
            - 1.5 * x3
            - 0.1 * x4 * cos(x4 + x5 - x1)
            + 0.2 * x5**2
            - x2
            - 1
        )


class DropWave(TestFunction):
    def __init__(self, dim=2):
        super(DropWave, self).__init__(dim)
        self.bounds = lzip([-2] * self.dim, [5.12] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = -1
        self.fmax = 0
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        norm_x = sum(x**2)
        return -(1 + cos(12 * sqrt(norm_x))) / (0.5 * norm_x + 2)


class Easom(TestFunction):
    def __init__(self, dim=2):
        super(Easom, self).__init__(dim)
        self.bounds = lzip([-100] * self.dim, [20] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 22.3504010789
        self.classifiers = ["unimodal", "boring"]

    def do_evaluate(self, x):
        a = 20
        b = 0.2
        c = 2 * pi
        n = self.dim
        return (
            -a * exp(-b * sqrt(sum(x**2) / n)) - exp(sum(cos(c * x)) / n) + a + exp(1)
        )


class EggCrate(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(EggCrate, self).__init__(dim)
        self.bounds = lzip([-5] * self.dim, [2] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 96.2896284292
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        x1, x2 = x
        return x1**2 + x2**2 + 25 * (sin(x1) ** 2 + sin(x2) ** 2)


class EggHolder(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(EggHolder, self).__init__(dim)
        self.bounds = lzip([-512.1] * self.dim, [512] * self.dim)
        self.min_loc = [512, 404.2319]
        self.fmin = -959.640662711
        self.fmax = 1049.53127276
        self.classifiers = ["bound_min"]

    def do_evaluate(self, x):
        x1, x2 = x
        return -(x2 + 47) * sin(sqrt(abs(x2 + x1 / 2 + 47))) - x1 * sin(
            sqrt(abs(x1 - (x2 + 47)))
        )


class ElAttarVidyasagarDutta(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(ElAttarVidyasagarDutta, self).__init__(dim)
        self.bounds = lzip([-100] * self.dim, [100] * self.dim)
        self.min_loc = [3.40918683, -2.17143304]
        self.fmin = 1.712780354
        self.fmax = 1.02030165675e12
        self.classifiers = ["unscaled"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (x1**2 + x2 - 10) ** 2 + (x1 + x2**2 - 7) ** 2 + (x1**2 + x2**3 - 1) ** 2


class Exponential(TestFunction):
    def __init__(self, dim=2):
        super(Exponential, self).__init__(dim)
        self.bounds = lzip([-0.7] * self.dim, [0.2] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = -1
        self.fmax = self.do_evaluate(asarray([-0.7] * self.dim))
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        return -exp(-0.5 * sum(x**2))


class Franke(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Franke, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.min_loc = [0.45571037432, 0.78419067287]
        self.fmin = 0.00111528244
        self.fmax = 1.22003257123

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            0.75 * exp(-((9 * x1 - 2) ** 2) / 4.0 - (9 * x2 - 2) ** 2 / 4.0)
            + 0.75 * exp(-((9 * x1 + 1) ** 2) / 49.0 - (9 * x2 + 1) / 10.0)
            + 0.5 * exp(-((9 * x1 - 7) ** 2) / 4.0 - (9 * x2 - 3) ** 2 / 4.0)
            - 0.2 * exp(-((9 * x1 - 4) ** 2) - (9 * x2 - 7) ** 2)
        )


class FreudensteinRoth(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(FreudensteinRoth, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [5, 4]
        self.fmin = 0
        self.fmax = 2908130
        self.classifiers = ["unscaled"]

    def do_evaluate(self, x):
        x1, x2 = x
        f1 = (-13 + x1 + ((5 - x2) * x2 - 2) * x2) ** 2
        f2 = (-29 + x1 + ((x2 + 1) * x2 - 14) * x2) ** 2
        return f1 + f2


class Gear(TestFunction):
    def __init__(self, dim=4):
        assert dim == 4
        super(Gear, self).__init__(dim)
        self.bounds = lzip([12] * self.dim, [60] * self.dim)
        self.min_loc = [16, 19, 43, 49]
        self.fmin = 2.7e-12
        self.fmax = 5
        self.classifiers = ["discrete", "multi_min", "boring", "complicated"]

    def do_evaluate(self, x):
        x1, x2, x3, x4 = x
        return min(
            (1 / 6.931 - floor(x1) * floor(x2) * 1.0 / (floor(x3) * floor(x4))) ** 2,
            self.fmax,
        )


class Giunta(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Giunta, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [1] * self.dim)
        self.min_loc = [0.4673200277395354, 0.4673200169591304]
        self.fmin = 0.06447042053690566
        self.fmax = 0.752651013458
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        arg = 16 * x / 15 - 1
        return 0.6 + sum(sin(arg) + sin(arg) ** 2 + sin(4 * arg) / 50)


class GoldsteinPrice(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(GoldsteinPrice, self).__init__(dim)
        self.bounds = lzip([-2] * self.dim, [2] * self.dim)
        self.min_loc = [0, -1]
        self.fmin = 3
        self.fmax = 1015689.58873
        self.classifiers = ["unscaled"]

    def do_evaluate(self, x):
        x1, x2 = x
        a = 1 + (x1 + x2 + 1) ** 2 * (
            19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        )
        b = 30 + (2 * x1 - 3 * x2) ** 2 * (
            18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        )
        return a * b


class Griewank(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Griewank, self).__init__(dim)
        self.bounds = lzip([-50] * self.dim, [20] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 3.187696592840877
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        return 1 + sum(x**2) / 4000 - prod(cos(x / sqrt(arange(1, self.dim + 1))))


class Hansen(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Hansen, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [-7.58989583, -7.70831466]
        self.fmin = -176.54
        self.fmax = 198.974631626
        self.classifiers = ["boring", "multi_min", "oscillatory"]

    def do_evaluate(self, x):
        x1, x2 = x
        return sum([(i + 1) * cos(i * x1 + i + 1) for i in range(5)]) * sum(
            [(i + 1) * cos((i + 2) * x2 + i + 1) for i in range(5)]
        )


class Hartmann3(TestFunction):
    def __init__(self, dim=3):
        assert dim == 3
        super(Hartmann3, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.min_loc = [0.1, 0.55592003, 0.85218259]
        self.fmin = -3.86278214782076
        self.fmax = -3.77271851416e-05

    def do_evaluate(self, x):
        a = asarray([[3, 0.1, 3, 0.1], [10, 10, 10, 10], [30, 35, 30, 35]])
        p = asarray(
            [
                [0.36890, 0.46990, 0.10910, 0.03815],
                [0.11700, 0.43870, 0.87320, 0.57430],
                [0.26730, 0.74700, 0.55470, 0.88280],
            ]
        )
        c = asarray([1, 1.2, 3, 3.2])
        d = zeros_like(c)
        for i in range(4):
            d[i] = sum(a[:, i] * (x - p[:, i]) ** 2)
        return -sum(c * exp(-d))


class Hartmann4(TestFunction):
    def __init__(self, dim=4):
        assert dim == 4
        super(Hartmann4, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.min_loc = [0.49204492762, 0.82366439640, 0.30064257056, 0.55643899079]
        self.fmin = -3.93518472715
        self.fmax = 1.31104361811

    def do_evaluate(self, x):
        a = asarray(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        p = asarray(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )
        c = asarray([1, 1.2, 3, 3.2])
        d = zeros_like(c)
        for i in range(4):
            d[i] = sum(a[:, i] * (x - p[:, i]) ** 2)
        return (1.1 - sum(c * exp(-d))) / 0.839


class Hartmann6(TestFunction):
    def __init__(self, dim=6):
        assert dim == 6
        super(Hartmann6, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.min_loc = [
            0.20168952,
            0.15001069,
            0.47687398,
            0.27533243,
            0.31165162,
            0.65730054,
        ]
        self.fmin = -3.32236801141551
        self.fmax = 0
        self.classifiers = ["boring"]

    def do_evaluate(self, x):
        a = asarray(
            [
                [10, 0.05, 3, 17],
                [3, 10, 3.5, 8],
                [17, 17, 1.7, 0.05],
                [3.5, 0.1, 10, 10],
                [1.7, 8, 17, 0.1],
                [8, 14, 8, 14],
            ]
        )
        p = asarray(
            [
                [0.1312, 0.2329, 0.2348, 0.4047],
                [0.1696, 0.4135, 0.1451, 0.8828],
                [0.5569, 0.8307, 0.3522, 0.8732],
                [0.0124, 0.3736, 0.2883, 0.5743],
                [0.8283, 0.1004, 0.3047, 0.1091],
                [0.5886, 0.9991, 0.6650, 0.0381],
            ]
        )
        c = asarray([1, 1.2, 3, 3.2])
        d = zeros_like(c)
        for i in range(4):
            d[i] = sum(a[:, i] * (x - p[:, i]) ** 2)
        return -sum(c * exp(-d))


class HelicalValley(TestFunction):
    def __init__(self, dim=3):
        assert dim == 3
        super(HelicalValley, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [2] * self.dim)
        self.min_loc = [1, 0, 0]
        self.fmin = 0
        self.fmax = 4902.295565
        self.classifiers = ["unscaled"]

    def do_evaluate(self, x):
        x1, x2, x3 = x
        return (
            100
            * (
                (x3 - 10 * arctan2(x2, x1) / 2 / pi) ** 2
                + (sqrt(x1**2 + x2**2) - 1) ** 2
            )
            + x3**2
        )


class HimmelBlau(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(HimmelBlau, self).__init__(dim)
        self.bounds = lzip([-2] * self.dim, [6] * self.dim)
        self.min_loc = [3, 2]
        self.fmin = 0
        self.fmax = 2186
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


class HolderTable(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(HolderTable, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [8.055023472141116, 9.664590028909654]
        self.fglob = -19.20850256788675
        self.fmin = -19.20850256788675
        self.fmax = 0
        self.classifiers = ["multi_min", "bound_min", "oscillatory", "complicated"]

    def do_evaluate(self, x):
        x1, x2 = x
        return -abs(sin(x1) * cos(x2) * exp(abs(1 - sqrt(x1**2 + x2**2) / pi)))


class Hosaki(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Hosaki, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [5] * self.dim)
        self.min_loc = [4, 2]
        self.fmin = -2.3458
        self.fmax = 0.54134113295

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            (1 + x1 * (-8 + x1 * (7 + x1 * (-2.33333 + x1 * 0.25))))
            * x2
            * x2
            * exp(-x2)
        )


class HosakiExpanded(Hosaki):
    def __init__(self, dim=2):
        assert dim == 2
        super(HosakiExpanded, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [10] * self.dim)
        self.fmax = 426.39606928
        self.classifiers = ["boring", "unscaled"]


class JennrichSampson(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(JennrichSampson, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [1] * self.dim)
        self.min_loc = [0.257825, 0.257825]
        self.fmin = 124.3621824
        self.fmax = 2241506295.39
        self.classifiers = ["boring", "unscaled"]

    def do_evaluate(self, x):
        x1, x2 = x
        rng = numpy.arange(10) + 1
        return sum((2 + 2 * rng - (exp(rng * x1) + exp(rng * x2))) ** 2)


class Judge(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Judge, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [0.86479, 1.2357]
        self.fmin = 16.0817307
        self.fmax = 58903.387568

    def do_evaluate(self, x):
        x1, x2 = x
        y_vec = asarray(
            [
                4.284,
                4.149,
                3.877,
                0.533,
                2.211,
                2.389,
                2.145,
                3.231,
                1.998,
                1.379,
                2.106,
                1.428,
                1.011,
                2.179,
                2.858,
                1.388,
                1.651,
                1.593,
                1.046,
                2.152,
            ]
        )
        x_vec = asarray(
            [
                0.286,
                0.973,
                0.384,
                0.276,
                0.973,
                0.543,
                0.957,
                0.948,
                0.543,
                0.797,
                0.936,
                0.889,
                0.006,
                0.828,
                0.399,
                0.617,
                0.939,
                0.784,
                0.072,
                0.889,
            ]
        )
        x_vec2 = asarray(
            [
                0.645,
                0.585,
                0.310,
                0.058,
                0.455,
                0.779,
                0.259,
                0.202,
                0.028,
                0.099,
                0.142,
                0.296,
                0.175,
                0.180,
                0.842,
                0.039,
                0.103,
                0.620,
                0.158,
                0.704,
            ]
        )
        return sum(((x1 + x2 * x_vec + (x2**2) * x_vec2) - y_vec) ** 2)


class Keane(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Keane, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [10] * self.dim)
        self.min_loc = [0, 1.39325]
        self.fmin = -0.67366751941
        self.fmax = 0
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        x1, x2 = x
        return -(sin(x1 - x2) ** 2 * sin(x1 + x2) ** 2) / sqrt(x1**2 + x2**2 + 1e-16)


class Langermann(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Langermann, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [10] * self.dim)
        self.min_loc = [2.00299219, 1.006096]
        self.fmin = -5.1621259
        self.fmax = 4.15526145026

    def do_evaluate(self, x):
        a = [3, 5, 2, 1, 7]
        b = [5, 2, 1, 4, 9]
        c = [1, 2, 5, 2, 3]
        x1, x2 = x
        return -sum(
            c
            * exp(-(1 / pi) * ((x1 - a) ** 2 + (x2 - b) ** 2))
            * cos(pi * ((x1 - a) ** 2 + (x2 - b) ** 2))
        )


class LennardJones6(TestFunction):
    def __init__(self, dim=6):
        assert dim == 6
        super(LennardJones6, self).__init__(dim)
        self.bounds = lzip([-3] * self.dim, [3] * self.dim)
        self.min_loc = [
            -2.66666470373,
            2.73904387714,
            1.42304625988,
            -1.95553276732,
            2.81714839844,
            2.12175295546,
        ]
        self.fmin = -1
        self.fmax = 0
        self.classifiers = ["boring", "multi_min"]

    def do_evaluate(self, x):
        k = int(self.dim / 3)
        s = 0
        for i in range(k - 1):
            for j in range(i + 1, k):
                a = 3 * i
                b = 3 * j
                xd = x[a] - x[b]
                yd = x[a + 1] - x[b + 1]
                zd = x[a + 2] - x[b + 2]
                ed = xd * xd + yd * yd + zd * zd
                ud = ed * ed * ed + 1e-8
                if ed > 0:
                    s += (1 / ud - 2) / ud
        return min(s, self.fmax)


class Leon(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Leon, self).__init__(dim)
        self.bounds = lzip([-1.2] * self.dim, [1.2] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 697
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        x1, x2 = x
        return 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2


class Levy03(TestFunction):
    def __init__(self, dim=8):
        assert dim == 8
        super(Levy03, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [1] * self.dim
        self.fmin = 0
        self.fmax = 573.929662663

    def do_evaluate(self, x):
        n = self.dim
        z = [1 + (xx - 1) / 4 for xx in x]
        s = sin(pi * z[0]) ** 2 + sum(
            [
                (z[i] - 1) ** 2 * (1 + 10 * (sin(pi * z[i] + 1)) ** 2)
                for i in range(n - 1)
            ]
        )
        return s + (z[n - 1] - 1) ** 2 * (1 + (sin(2 * pi * z[n - 1])) ** 2)


class Levy05(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Levy05, self).__init__(dim)
        self.bounds = lzip([-2] * self.dim, [2] * self.dim)
        self.min_loc = [-0.34893137569, -0.79113519694]
        self.fmin = -135.27125929718
        self.fmax = 244.97862255137
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        x1, x2 = x
        rng = numpy.arange(5) + 1
        return (
            sum(rng * cos((rng - 1) * x1 + rng)) * sum(rng * cos((rng + 1) * x2 + rng))
            + (x1 * 5 + 1.42513) ** 2
            + (x2 * 5 + 0.80032) ** 2
        )


class Levy13(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Levy13, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [1] * self.dim
        self.fmin = 0
        self.fmax = 454.12864891174
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            (sin(3 * pi * x1)) ** 2
            + ((x1 - 1) ** 2) * (1 + (sin(3 * pi * x2)) ** 2)
            + ((x2 - 1) ** 2) * (1 + (sin(2 * pi * x2)) ** 2)
        )


class Matyas(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Matyas, self).__init__(dim)
        self.bounds = [[-10, 3], [-3, 10]]
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 44.09847214410

    def do_evaluate(self, x):
        x1, x2 = x
        return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


class McCormick(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        TestFunction.__init__(self, dim)
        self.bounds = [(-1.5, 4), (-3, 4)]
        self.min_loc = [-0.5471975602214493, -1.547197559268372]
        self.fmin = -1.913222954981037
        self.fmax = 44.0984721441

    def do_evaluate(self, x):
        x1, x2 = x
        return sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1


class McCourtBase(TestFunction):
    """
    This is a class of functions that all fit into the framework of a linear combination of functions, many of
    which are positive definite kernels, but not all.

    These were created by playing around with parameter choices for long enough until a function with desired
    properties was produced.
    """

    @staticmethod
    def dist_sq(x, centers, e_mat, dist_type=2):
        if dist_type == 1:
            ret_val = numpy.array(
                [
                    [
                        numpy.sum(numpy.abs((xpt - center) * evec))
                        for evec, center in lzip(numpy.sqrt(e_mat), centers)
                    ]
                    for xpt in x
                ]
            )
        elif dist_type == 2:
            ret_val = numpy.array(
                [
                    [
                        numpy.dot((xpt - center) * evec, (xpt - center))
                        for evec, center in lzip(e_mat, centers)
                    ]
                    for xpt in x
                ]
            )
        elif dist_type == "inf":
            ret_val = numpy.array(
                [
                    [
                        numpy.max(numpy.abs((xpt - center) * evec))
                        for evec, center in lzip(numpy.sqrt(e_mat), centers)
                    ]
                    for xpt in x
                ]
            )
        else:
            raise ValueError("Unrecognized distance type {0}".format(dist_type))
        return ret_val

    def __init__(self, dim, kernel, e_mat, coefs, centers):
        super(McCourtBase, self).__init__(dim)
        assert e_mat.shape == centers.shape
        assert e_mat.shape[0] == coefs.shape[0]
        assert e_mat.shape[1] == dim
        self.kernel = kernel
        self.e_mat = e_mat
        self.coefs = coefs
        self.centers = centers
        self.bounds = [(0, 1)] * dim

    def do_evaluate(self, x):
        return_1d = False
        if len(x.shape) == 1:  # In case passed as a single vector instead of 2D array
            x = x[numpy.newaxis, :]
            return_1d = True
        assert self.e_mat.shape[1] == x.shape[1]  # Correct dimension
        ans = numpy.sum(self.coefs * self.kernel(x), axis=1)
        return ans[0] if return_1d else ans


class McCourt01(McCourtBase):
    def __init__(self, dim=7):
        assert dim == 7
        centers = numpy.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.3, 0.1, 0.5, 0.1, 0.8, 0.8, 0.6],
                [0.6, 0.7, 0.8, 0.3, 0.7, 0.8, 0.6],
                [0.4, 0.7, 0.4, 0.9, 0.4, 0.1, 0.9],
                [0.9, 0.3, 0.3, 0.5, 0.2, 0.7, 0.2],
                [0.5, 0.5, 0.2, 0.8, 0.5, 0.3, 0.4],
            ]
        )
        e_mat = 5 * numpy.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        )
        coefs = numpy.array([1, 1, -2, 1, 1, 1])

        def kernel(x):
            r2 = self.dist_sq(x, centers, e_mat)
            return 1 / numpy.sqrt(1 + r2)

        super(McCourt01, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.6241, 0.7688, 0.8793, 0.2739, 0.7351, 0.8499, 0.6196]
        self.fmin = -0.0859426686096
        self.fmax = 2.06946125482978


class McCourt02(McCourtBase):
    def __init__(self, dim=7):
        assert dim == 7
        centers = numpy.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.3, 0.1, 0.5, 0.1, 0.8, 0.8, 0.6],
                [0.6, 0.7, 0.8, 0.3, 0.7, 0.8, 0.6],
                [0.4, 0.7, 0.4, 0.9, 0.4, 0.1, 0.9],
                [0.9, 0.3, 0.3, 0.5, 0.2, 0.7, 0.2],
                [0.5, 0.5, 0.2, 0.8, 0.5, 0.3, 0.4],
            ]
        )
        e_mat = 5 * numpy.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        )
        coefs = numpy.array([-1, -1, -2, 1, 1, -1])

        def kernel(x):
            r2 = self.dist_sq(x, centers, e_mat)
            return 1 / numpy.sqrt(1 + r2)

        super(McCourt02, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.4068, 0.4432, 0.6479, 0.1978, 0.7660, 0.7553, 0.5640]
        self.fmin = -2.74162116801
        self.fmax = -1.25057003098


class McCourt03(McCourtBase):
    def __init__(self, dim=9):
        assert dim == 9
        centers = numpy.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.3, 0.1, 0.5, 0.1, 0.8, 0.8, 0.6, 0.4, 0.2],
                [0.6, 0.7, 0.8, 0.3, 0.7, 0.8, 0.6, 0.9, 0.1],
                [0.7, 0.2, 0.7, 0.7, 0.3, 0.3, 0.8, 0.6, 0.4],
                [0.4, 0.6, 0.4, 0.9, 0.4, 0.1, 0.9, 0.3, 0.3],
                [0.5, 0.5, 0.2, 0.8, 0.5, 0.3, 0.4, 0.5, 0.8],
                [0.8, 0.3, 0.3, 0.5, 0.2, 0.7, 0.2, 0.4, 0.6],
                [0.8, 0.3, 0.3, 0.5, 0.2, 0.7, 0.2, 0.4, 0.6],
                [0.8, 0.3, 0.3, 0.5, 0.2, 0.7, 0.2, 0.4, 0.6],
            ]
        )
        e_mat = numpy.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        coefs = numpy.array([1, -1, 1, 1, 1, 1, -1, -2, -1])

        def kernel(x):
            r2 = self.dist_sq(x, centers, e_mat)
            return numpy.exp(-r2)

        super(McCourt03, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [
            0.9317,
            0.1891,
            0.2503,
            0.3646,
            0.1603,
            0.9829,
            0.0392,
            0.3263,
            0.6523,
        ]
        self.fmin = -3.02379637466
        self.fmax = 0.28182628628


class McCourt04(McCourtBase):
    def __init__(self, dim=10):
        assert dim == 10
        centers = numpy.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.3, 0.1, 0.5, 0.1, 0.8, 0.8, 0.6, 0.4, 0.2, 0.9],
                [0.6, 0.7, 0.8, 0.3, 0.7, 0.8, 0.6, 0.9, 0.1, 0.2],
                [0.7, 0.2, 0.7, 0.7, 0.3, 0.3, 0.8, 0.6, 0.4, 0.1],
                [0.4, 0.6, 0.4, 0.9, 0.4, 0.1, 0.9, 0.3, 0.3, 0.2],
                [0.5, 0.5, 0.2, 0.8, 0.5, 0.3, 0.4, 0.5, 0.8, 0.6],
                [0.8, 0.4, 0.3, 0.5, 0.2, 0.7, 0.2, 0.4, 0.6, 0.5],
                [0.8, 0.4, 0.3, 0.5, 0.2, 0.7, 0.2, 0.4, 0.6, 0.5],
                [0.8, 0.4, 0.3, 0.5, 0.2, 0.7, 0.2, 0.4, 0.6, 0.5],
            ]
        )
        e_mat = 0.5 * numpy.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        coefs = numpy.array([1, -1, 1, -1, 1, 1, -2, -1, -1])

        def kernel(x):
            r2 = self.dist_sq(x, centers, e_mat)
            return numpy.cos(numpy.pi * numpy.sqrt(r2)) * numpy.exp(-r2)

        super(McCourt04, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [
            0.8286,
            0.3562,
            0.3487,
            0.4623,
            0.1549,
            0.7182,
            0.2218,
            0.3919,
            0.5394,
            0.441,
        ]
        self.fmin = -4.631135472012
        self.fmax = 0.81136346883


class McCourt05(McCourtBase):
    def __init__(self, dim=12):
        assert dim == 12
        centers = numpy.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.3, 0.1, 0.5, 0.1, 0.8, 0.8, 0.6, 0.4, 0.2, 0.9, 0.3, 0.7],
                [0.6, 0.7, 0.8, 0.3, 0.7, 0.8, 0.6, 0.9, 0.1, 0.2, 0.5, 0.2],
                [0.7, 0.2, 0.7, 0.7, 0.3, 0.3, 0.8, 0.6, 0.4, 0.1, 0.9, 0.9],
                [0.4, 0.6, 0.4, 0.5, 0.4, 0.2, 0.8, 0.3, 0.3, 0.2, 0.5, 0.1],
                [0.5, 0.5, 0.2, 0.8, 0.5, 0.3, 0.4, 0.5, 0.8, 0.6, 0.9, 0.1],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 0.1, 0.2],
                [0.8, 0.4, 0.3, 0.5, 0.2, 0.7, 0.2, 0.4, 0.6, 0.5, 0.3, 0.8],
                [0.9, 0.5, 0.3, 0.2, 0.1, 0.9, 0.3, 0.7, 0.7, 0.7, 0.4, 0.4],
                [0.2, 0.8, 0.6, 0.4, 0.6, 0.6, 0.5, 0, 0.2, 0.8, 0.2, 0.3],
            ]
        )
        e_mat = 0.4 * numpy.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        coefs = numpy.array([5, -2, 5, -5, -20, -2, 10, 2, -5, 5])

        def kernel(x):
            r2 = self.dist_sq(x, centers, e_mat)
            return numpy.exp(-r2)

        super(McCourt05, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [
            0.636,
            0.622,
            0.39,
            0.622,
            0.29,
            0.047,
            0.97,
            0.26,
            0.311,
            0.247,
            0.794,
            0.189,
        ]
        self.fmin = -11.89842508364
        self.fmax = 2.821916955234


class McCourt06(McCourtBase):
    def __init__(self, dim=5):
        assert dim == 5
        centers = numpy.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1],
                [0.3, 0.8, 0.8, 0.6, 0.9],
                [0.6, 0.1, 0.2, 0.5, 0.2],
                [0.7, 0.2, 0.1, 0.8, 0.9],
                [0.4, 0.6, 0.5, 0.3, 0.8],
                [0.9, 0.5, 0.3, 0.2, 0.4],
                [0.2, 0.8, 0.6, 0.4, 0.6],
            ]
        )
        e_mat = 0.4 * numpy.array(
            [
                [1, 1, 1, 1, 1],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        coefs = numpy.array([-3, 2, -2, 4, -1, 5, -1])

        def kernel(x):
            r2 = self.dist_sq(x, centers, e_mat)
            return numpy.sqrt(1 + r2)

        super(McCourt06, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [1, 1, 0.7636, 0.5268, 1]
        self.fmin = 2.80720263234
        self.fmax = 5.26036468689
        self.classifiers = ["bound_min"]


class McCourt07(McCourtBase):
    def __init__(self, dim=6):
        assert dim == 6
        centers = numpy.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.3, 0.8, 0.8, 0.6, 0.9, 0.4],
                [0.6, 1, 0.2, 0, 1, 0.3],
                [0.7, 0.2, 0.1, 0.8, 0.9, 0.2],
                [0.4, 0.6, 0.5, 0.3, 0.8, 0.3],
                [0.9, 0.5, 0.3, 0.2, 0.4, 0.8],
                [0.2, 0.8, 0.6, 0.4, 0.6, 0.9],
            ]
        )
        e_mat = 0.7 * numpy.array(
            [
                [1, 1, 1, 1, 1, 1],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1],
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1],
                [0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
            ]
        )
        coefs = numpy.array([2, 2, -4, 1, -2, 4, -2])

        def kernel(x):
            r = numpy.sqrt(self.dist_sq(x, centers, e_mat))
            return (1 + r) * numpy.exp(-r)

        super(McCourt07, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.3811, 1, 0.2312, 0, 1, 0.1403]
        self.fglob = -0.36321372933
        self.fmin = -0.36321372933
        self.fmax = 1.86724590652
        self.classifiers = ["bound_min", "nonsmooth"]


class McCourt08(McCourtBase):
    def __init__(self, dim=4):
        assert dim == 4
        centers = numpy.array(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.3, 0.8, 0.9, 0.4],
                [0.6, 1, 0.2, 0],
                [0.7, 0.2, 0.1, 0.8],
                [0.4, 0, 0.8, 1],
                [0.9, 0.5, 0.3, 0.2],
                [0.2, 0.8, 0.6, 0.4],
            ]
        )
        e_mat = 0.7 * numpy.array(
            [
                [1, 1, 1, 1],
                [0.5, 0.5, 0.5, 0.5],
                [1, 3, 1, 3],
                [0.5, 0.5, 0.5, 0.5],
                [2, 1, 2, 1],
                [1, 1, 1, 1],
                [0.7, 0.7, 0.7, 0.7],
            ]
        )
        coefs = numpy.array([2, 1, -8, 1, -5, 3, 2])

        def kernel(x):
            r = numpy.sqrt(self.dist_sq(x, centers, e_mat))
            return (1 + r + 0.333 * r**2) * numpy.exp(-r)

        super(McCourt08, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.5067, 1, 0.5591, 0.0823]
        self.fmin = -3.45224058874
        self.fmax = -0.60279774058
        self.classifiers = ["bound_min", "nonsmooth"]


class McCourt09(McCourtBase):
    def __init__(self, dim=3):
        assert dim == 3
        centers = numpy.array(
            [
                [0.1, 0.1, 0.1],
                [0.3, 0.8, 0.9],
                [0.6, 1, 0.2],
                [0.6, 1, 0.2],
                [0.7, 0.2, 0.1],
                [0.4, 0, 0.8],
                [0.9, 0.5, 1],
                [0, 0.8, 0.6],
            ]
        )
        e_mat = 0.6 * numpy.array(
            [
                [1, 1, 1],
                [0.6, 0.6, 0.6],
                [1, 0.5, 1],
                [4, 10, 4],
                [0.5, 0.5, 0.5],
                [0.5, 1, 0.5],
                [1, 1, 1],
                [0.3, 0.5, 0.5],
            ]
        )
        coefs = numpy.array([4, -3, -6, -2, 1, -3, 6, 2])

        def kernel(x):
            r2 = self.dist_sq(x, centers, e_mat)
            return numpy.cos(numpy.pi * numpy.sqrt(r2)) * numpy.exp(-r2)

        super(McCourt09, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.594, 1, 0.205]
        self.fmin = -10.17146707797
        self.fmax = 6.55195724520
        self.classifiers = ["bound_min"]


class McCourt10(McCourtBase):
    def __init__(self, dim=8):
        assert dim == 8
        centers = numpy.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.3, 0.1, 0.5, 0.1, 0.8, 0.8, 0.6, 0.4],
                [0.6, 0.7, 0.8, 0.3, 0.7, 0.8, 0.6, 0.9],
                [0.7, 0, 0.7, 1, 0.3, 0, 0.8, 0.6],
                [0.4, 0.6, 0.4, 1, 0.4, 0.2, 1, 0.3],
                [0.5, 0.5, 0.2, 0.8, 0.5, 0.3, 0.4, 0.5],
                [0.1, 0.2, 1, 0.4, 0.5, 0.6, 0.7, 0],
                [0.9, 0.4, 0.3, 0.5, 0.2, 0.7, 0.2, 0.4],
                [0, 0.5, 0.3, 0.2, 0.1, 0.9, 0.3, 0.7],
                [0.2, 0.8, 0.6, 0.4, 0.6, 0.6, 0.5, 0],
            ]
        )
        e_mat = 0.8 * numpy.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [3, 3, 3, 3, 3, 3, 3, 3],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        coefs = numpy.array([5, -2, 5, -5, -12, -2, 10, 2, -5, 5])

        def kernel(x):
            r2 = self.dist_sq(x, centers, e_mat)
            return 1 / numpy.sqrt(1 + r2)

        super(McCourt10, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.5085, 0.5433, 0.2273, 1, 0.3381, 0.0255, 1, 0.5038]
        self.fmin = -2.51939597030
        self.fmax = 5.81472085012
        self.classifiers = ["bound_min"]


class McCourt11(McCourtBase):
    def __init__(self, dim=8):
        assert dim == 8
        centers = numpy.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.3, 0.1, 0.5, 0.1, 0.8, 0.8, 0.6, 0.4],
                [0.6, 0.7, 0.8, 0.3, 0.7, 0.8, 0.6, 0.9],
                [0.7, 0, 0.7, 1, 0.3, 0, 0.8, 0.6],
                [0.4, 0.6, 0.4, 1, 0.4, 0.2, 1, 0.3],
                [0.5, 0.5, 0.2, 0.8, 0.5, 0.3, 0.4, 0.5],
                [0.1, 0.2, 1, 0.4, 0.5, 0.6, 0.7, 0],
                [0.9, 0.4, 0.3, 0.5, 0.2, 0.7, 0.2, 0.4],
                [0, 0.5, 0.3, 0.2, 0.1, 0.9, 0.3, 0.7],
                [0.2, 0.8, 0.6, 0.4, 0.6, 0.6, 0.5, 0],
            ]
        )
        e_mat = 0.5 * numpy.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [3, 3, 3, 3, 3, 3, 3, 3],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        coefs = numpy.array([5, -2, 5, -5, -7, -2, 10, 2, -5, 5])

        def kernel(x):
            r = numpy.sqrt(self.dist_sq(x, centers, e_mat))
            return numpy.exp(-r)

        super(McCourt11, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.4, 0.6, 0.4, 1, 0.4, 0.2, 1, 0.3]
        self.fmin = -0.39045528652
        self.fmax = 9.07754532532
        self.classifiers = ["bound_min", "nonsmooth"]


class McCourt12(McCourtBase):
    def __init__(self, dim=7):
        assert dim == 7
        centers = numpy.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.3, 0.1, 0.5, 0.1, 0.8, 0.8, 0.6],
                [0.6, 0.7, 0.8, 0.3, 0.7, 0.8, 0.6],
                [0.7, 0, 0.7, 1, 0.3, 0, 0.8],
                [0.4, 0.6, 0.4, 1, 0.4, 0.2, 1],
                [0.5, 0.5, 0.2, 0.8, 0.5, 0.3, 0.4],
                [0.1, 0.2, 1, 0.4, 0.5, 0.6, 0.7],
                [0.9, 0.4, 0.3, 0.5, 0.2, 0.7, 0.2],
                [0, 0.5, 0.3, 0.2, 0.1, 0.9, 0.3],
                [0.2, 0.8, 0.6, 0.4, 0.6, 0.6, 0.5],
            ]
        )
        e_mat = 0.7 * numpy.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1, 1],
                [10, 10, 10, 10, 10, 10, 10],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        )
        coefs = numpy.array([5, -4, 5, -5, -7, -2, 10, 2, -5, 5])

        def kernel(x):
            r = numpy.sqrt(self.dist_sq(x, centers, e_mat))
            return besselj(0, r)

        super(McCourt12, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.4499, 0.4553, 0.0046, 1, 0.3784, 0.3067, 0.6173]
        self.fmin = 3.54274987790
        self.fmax = 9.92924222433
        self.classifiers = ["bound_min", "oscillatory"]


class McCourt13(McCourtBase):
    def __init__(self, dim=3):
        assert dim == 3
        centers = numpy.array(
            [
                [0.9, 0.9, 0.9],
                [0.9, 0.9, 1],
                [0.9, 1, 0.9],
                [1, 0.9, 0.9],
                [1, 1, 1],
                [1, 0, 0],
                [0.5, 0, 0],
                [0, 1, 0],
                [0, 0.7, 0],
                [0, 0, 0],
                [0.4, 0.3, 0.6],
                [0.7, 0.7, 0.7],
                [0.7, 0.7, 1],
                [1, 0.7, 0.7],
                [0.7, 1, 0.7],
            ]
        )
        e_mat = 0.8 * numpy.array(
            [
                [9.5, 9.5, 9.5],
                [9.5, 9.5, 9.5],
                [9.5, 9.5, 9.5],
                [9.5, 9.5, 9.5],
                [9.5, 9.5, 9.5],
                [1, 0.5, 1],
                [2, 0.5, 1],
                [0.5, 0.5, 0.5],
                [0.5, 1, 0.5],
                [1, 1, 1],
                [2, 2, 3.5],
                [8.5, 8.5, 8.5],
                [8.5, 8.5, 8.5],
                [8.5, 8.5, 8.5],
                [8.5, 8.5, 8.5],
            ]
        )
        coefs = numpy.array([4, 4, 4, 4, -12, 1, 3, -2, 5, -2, 1, -2, -2, -2, -2])

        def kernel(x):
            r2 = self.dist_sq(x, centers, e_mat)
            return numpy.exp(-r2)

        super(McCourt13, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [1, 1, 1]
        self.fmin = 1.49048296359
        self.fmax = 5.15444049449
        self.classifiers = ["bound_min"]


class McCourt14(McCourtBase):
    def __init__(self, dim=3):
        assert dim == 3
        centers = numpy.array(
            [
                [0.1, 0.8, 0.3],
            ]
        )
        e_mat = numpy.array(
            [
                [5, 5, 5],
            ]
        )
        coefs = numpy.array([-5])

        def kernel(x):
            r2 = self.dist_sq(x, centers, e_mat)
            return numpy.exp(-r2)

        super(McCourt14, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.1, 0.8, 0.3]
        self.fmin = -5
        self.fmax = 0.00030641748
        self.classifiers = ["boring", "unimodal"]


class McCourt15(McCourtBase):
    def __init__(self, dim=3):
        assert dim == 3
        centers = numpy.array(
            [
                [0.1, 0.8, 0.3],
            ]
        )
        e_mat = numpy.array(
            [
                [7, 7, 7],
            ]
        )
        coefs = numpy.array([-5])

        def kernel(x):
            r = numpy.sqrt(self.dist_sq(x, centers, e_mat))
            return numpy.exp(-r)

        super(McCourt15, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.1, 0.8, 0.3]
        self.fmin = -5
        self.fmax = 0.00030641748
        self.classifiers = ["boring", "unimodal", "nonsmooth"]


class McCourt16(McCourtBase):
    def __init__(self, dim=4):
        assert dim == 4
        centers = numpy.array(
            [
                [0.3, 0.8, 0.3, 0.6],
                [0.4, 0.9, 0.4, 0.7],
            ]
        )
        e_mat = numpy.array(
            [
                [5, 5, 5, 5],
                [5, 5, 5, 5],
            ]
        )
        coefs = numpy.array([-5, 5])

        def kernel(x):
            r2 = self.dist_sq(x, centers, e_mat)
            return 1 / numpy.sqrt(1 + r2)

        super(McCourt16, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.1858, 0.6858, 0.1858, 0.4858]
        self.fmin = -0.84221700966
        self.fmax = 0.84132432380
        self.classifiers = ["boring", "unimodal"]


class McCourt17(McCourtBase):
    def __init__(self, dim=7):
        assert dim == 7
        centers = numpy.array(
            [
                [0.3, 0.8, 0.3, 0.6, 0.2, 0.8, 0.5],
                [0.8, 0.3, 0.8, 0.2, 0.5, 0.2, 0.8],
                [0.2, 0.7, 0.2, 0.5, 0.4, 0.7, 0.3],
            ]
        )
        e_mat = numpy.array(
            [
                [4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4],
            ]
        )
        coefs = numpy.array([-5, 5, 5])

        def kernel(x):
            r2 = self.dist_sq(x, centers, e_mat)
            return 1 / numpy.sqrt(1 + r2)

        super(McCourt17, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.3125, 0.9166, 0.3125, 0.7062, 0.0397, 0.9270, 0.5979]
        self.fmin = -0.47089199032
        self.fmax = 4.98733340158
        self.classifiers = ["boring", "unimodal"]


class McCourt18(McCourtBase):
    def __init__(self, dim=8):
        assert dim == 8
        centers = numpy.array(
            [
                [0.3, 0.8, 0.3, 0.6, 0.2, 0.8, 0.2, 0.4],
                [0.3, 0.8, 0.3, 0.6, 0.2, 0.8, 0.2, 0.4],
                [0.3, 0.8, 0.3, 0.6, 0.2, 0.8, 0.2, 0.4],
                [0.8, 0.3, 0.8, 0.2, 0.5, 0.2, 0.5, 0.7],
                [0.2, 0.7, 0.2, 0.5, 0.4, 0.3, 0.8, 0.8],
            ]
        )
        e_mat = numpy.array(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
            ]
        )
        coefs = numpy.array([-1, 2, -5, 4, 4])

        def kernel(x):
            r = numpy.sqrt(self.dist_sq(x, centers, e_mat))
            return (1 + r) * exp(-r)

        super(McCourt18, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.2677, 0.8696, 0.2677, 0.6594, 0.1322, 0.9543, 0.0577, 0.295]
        self.fmin = -1.42906223657
        self.fmax = 4.76974923199
        self.classifiers = ["boring", "nonsmooth"]


class McCourt19(McCourtBase):
    def __init__(self, dim=2):
        assert dim == 2
        centers = numpy.array(
            [
                [0.1, 0.1],
                [0.3, 0.8],
                [0.6, 0.7],
                [0.7, 0.1],
                [0.4, 0.3],
                [0.2, 0.8],
                [0.1, 0.2],
                [0.9, 0.4],
                [0.5, 0.5],
                [0, 0.8],
            ]
        )
        e_mat = 3 * numpy.array(
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [0.5, 0.5],
                [1, 1],
                [3, 3],
                [0.5, 0.5],
                [1, 1],
                [2, 2],
                [1, 1],
            ]
        )
        coefs = -numpy.array([5, -4, 5, -5, -4, -2, 10, 4, -5, -5])

        def kernel(x):
            rabs = self.dist_sq(x, centers, e_mat, dist_type=1)
            return rabs

        super(McCourt19, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.4, 0.8]
        self.fmin = -8.67263950474
        self.fmax = 21.39025479756
        self.classifiers = ["nonsmooth"]


class McCourt20(McCourtBase):
    def __init__(self, dim=2):
        assert dim == 2
        centers = numpy.array(
            [
                [0.1, 0.1],
                [0.3, 0.8],
                [0.6, 0.7],
                [0.7, 0.1],
                [0.4, 0.3],
                [0.2, 0.8],
                [0.1, 0.2],
                [0.9, 0.4],
                [0.5, 0.5],
                [0, 0.8],
            ]
        )
        e_mat = 50 * numpy.array(
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [0.5, 0.5],
                [1, 1],
                [3, 3],
                [0.5, 0.5],
                [1, 1],
                [2, 2],
                [1, 1],
            ]
        )
        coefs = numpy.array([5, -4, 5, -7, -4, -2, 10, 4, -2, -5])

        def kernel(x):
            rabs = self.dist_sq(x, centers, e_mat, dist_type=1)
            return numpy.exp(-rabs)

        super(McCourt20, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.7, 0.1]
        self.fmin = -6.59763663216
        self.fmax = 11.97358068925
        self.classifiers = ["nonsmooth"]


class McCourt21(McCourtBase):
    def __init__(self, dim=4):
        assert dim == 4
        centers = numpy.array(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.3, 0.8, 0.5, 0.2],
                [0, 0.7, 0.4, 0.9],
                [0.7, 0.1, 0.2, 0.8],
                [0.4, 0.3, 0.6, 0.6],
                [0.2, 0.8, 0.2, 0.6],
                [0.9, 0.2, 0.3, 0.4],
                [0.9, 0.4, 0.9, 0.8],
                [0.5, 0.5, 0.5, 0.5],
                [0, 0.8, 0, 0.2],
            ]
        )
        e_mat = 10 * numpy.array(
            [
                [1, 1, 4, 4],
                [1, 1, 4, 4],
                [3, 3, 4, 4],
                [0.5, 0.5, 2, 2],
                [1, 1, 0.5, 0.2],
                [3, 3, 1, 1],
                [0.5, 0.5, 4, 2],
                [1, 1, 2, 3],
                [2, 2, 3, 4],
                [1, 1, 0.5, 0.5],
            ]
        )
        coefs = numpy.array([5, -4, 5, -5, 4, -2, 10, -8, -2, -5])

        def kernel(x):
            rmax = self.dist_sq(x, centers, e_mat, dist_type="inf")
            return numpy.exp(-rmax)

        super(McCourt21, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.9, 0.4, 0.9, 0.8]
        self.fmin = -7.74993665759
        self.fmax = 8.31973328564
        self.classifiers = ["nonsmooth"]


class McCourt22(McCourtBase):
    def __init__(self, dim=5):
        assert dim == 5
        centers = numpy.array(
            [
                [1, 0.3, 0.1, 0.4, 0.1],
                [0.9, 0.7, 0, 0.5, 0.8],
                [0.5, 0.6, 0.6, 0.5, 0.5],
                [0.2, 0.2, 0.4, 0, 0.3],
                [0, 0.6, 1, 0.1, 0.8],
                [0.3, 0.5, 0.8, 0, 0.2],
                [0.8, 1, 0.1, 0.1, 0.5],
            ]
        )
        e_mat = 5 * numpy.array(
            [
                [1, 6, 5, 1, 3],
                [2, 6, 2, 1, 1],
                [1, 2, 1, 2, 1],
                [4, 1, 4, 1, 1],
                [5, 6, 1, 3, 2],
                [4, 2, 3, 1, 4],
                [3, 5, 1, 4, 5],
            ]
        )
        coefs = numpy.array([3, 4, -4, 2, -3, -2, 6])

        def kernel(x):
            rmax = self.dist_sq(x, centers, e_mat, dist_type="inf")
            return numpy.exp(-rmax)

        super(McCourt22, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.2723, 0.4390, 0.8277, 0.3390, 0.3695]
        self.fmin = -3.08088199150
        self.fmax = 4.96977632014
        self.classifiers = ["nonsmooth"]


class McCourt23(McCourtBase):
    def __init__(self, dim=6):
        assert dim == 6
        centers = numpy.array(
            [
                [0.1, 0.1, 1, 0.3, 0.4, 0.1],
                [0, 0, 0.1, 0.6, 0, 0.7],
                [0.1, 0.5, 0.7, 0, 0.7, 0.3],
                [0.9, 0.6, 0.2, 0.9, 0.3, 0.8],
                [0.8, 0.3, 0.7, 0.7, 0.2, 0.7],
                [0.7, 0.6, 0.5, 1, 1, 0.7],
                [0.8, 0.9, 0.5, 0, 0, 0.5],
                [0.3, 0, 0.3, 0.2, 0.1, 0.8],
            ]
        )
        e_mat = 0.1 * numpy.array(
            [
                [4, 5, 5, 4, 1, 5],
                [2, 4, 5, 1, 2, 2],
                [1, 4, 3, 2, 2, 3],
                [4, 2, 3, 4, 1, 4],
                [2, 3, 6, 6, 4, 1],
                [5, 4, 1, 4, 1, 1],
                [2, 2, 2, 5, 4, 2],
                [1, 4, 6, 3, 4, 3],
            ]
        )
        coefs = numpy.array([1, -2, 3, -20, 5, -2, -1, -2])

        def kernel(x):
            rmax = self.dist_sq(x, centers, e_mat, dist_type="inf")
            return besselj(0, rmax)

        super(McCourt23, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.7268, 0.3914, 0, 0.7268, 0.5375, 0.8229]
        self.fmin = -18.35750245671
        self.fmax = -16.07462900440
        self.classifiers = ["nonsmooth", "bound_min"]


class McCourt24(McCourtBase):
    def __init__(self, dim=7):
        assert dim == 7
        centers = numpy.array(
            [
                [0, 0.4, 0, 0.3, 0.2, 0.3, 0.6],
                [0.6, 0.8, 0.6, 0.7, 0.7, 0.1, 0.4],
                [0.7, 0.7, 0, 0.5, 0, 0.6, 0.8],
                [0.7, 0.5, 0.6, 0.2, 0.5, 0.3, 0.2],
                [0.9, 0.3, 0.9, 0.8, 0.7, 1, 0],
                [0.8, 0.1, 0.1, 0.2, 0.6, 0.1, 0.3],
                [0.2, 0.7, 0.5, 0.5, 1, 0.7, 0.4],
                [0.4, 0.1, 0.4, 0.1, 0.9, 0.2, 0.9],
                [0.6, 0.9, 0.1, 0.4, 0.8, 0.7, 0.1],
            ]
        )
        e_mat = 0.2 * numpy.array(
            [
                [1, 2, 2, 3, 5, 2, 1],
                [5, 2, 3, 3, 4, 2, 4],
                [5, 4, 2, 1, 4, 1, 4],
                [4, 1, 2, 5, 1, 2, 5],
                [2, 4, 4, 4, 5, 5, 3],
                [1, 2, 5, 2, 1, 4, 6],
                [1, 6, 2, 1, 4, 5, 6],
                [1, 1, 5, 1, 4, 5, 5],
                [3, 5, 1, 3, 2, 5, 4],
            ]
        )
        coefs = numpy.array([1, 2, 3, -4, 3, -2, -1, -2, 5])

        def kernel(x):
            rmax = self.dist_sq(x, centers, e_mat, dist_type=1)
            return 1 / (1 + rmax)

        super(McCourt24, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.7, 0.1369, 0.6, 0.2, 0.5, 0.3, 0.2]
        self.fmin = -0.17296443752
        self.fmax = 4.98299597248
        self.classifiers = ["nonsmooth"]


class McCourt25(McCourtBase):  # Need fixed somehow
    def __init__(self, dim=8):
        assert dim == 8
        centers = numpy.array(
            [
                [0.5, 0, 0.3, 0.5, 0.8, 0.3, 0.2, 1],
                [0.6, 0.1, 0.6, 0.9, 0.2, 0, 0.5, 0.9],
                [0.9, 0.9, 0, 1, 0.5, 1, 0.1, 0],
                [0.2, 0.6, 0.4, 0.8, 0.4, 0.3, 0.9, 0.8],
                [0.2, 0.8, 0.5, 0.1, 0.7, 0.2, 0.4, 0.8],
                [0.2, 0.1, 0.7, 0.6, 0.2, 1, 0.6, 0.2],
                [0.5, 0.8, 0.6, 0, 0.6, 0.3, 0.3, 0.2],
                [0, 0, 0.2, 0.8, 0.9, 0.1, 0.1, 0.5],
                [0.9, 0.9, 0.1, 0.3, 0.9, 0.8, 0.7, 0],
                [0.3, 0.2, 0.9, 0.8, 0.9, 0.3, 0, 0.7],
            ]
        )
        e_mat = 5 * numpy.array(
            [
                [5, 4, 4, 6, 4, 5, 3, 1],
                [6, 6, 1, 5, 2, 5, 3, 2],
                [2, 4, 5, 2, 3, 6, 5, 2],
                [2, 1, 3, 2, 1, 1, 2, 4],
                [4, 3, 6, 4, 1, 1, 5, 4],
                [5, 1, 6, 1, 4, 6, 4, 6],
                [5, 3, 3, 3, 1, 3, 4, 5],
                [5, 4, 2, 5, 1, 5, 3, 5],
                [6, 4, 2, 1, 1, 5, 5, 4],
                [3, 3, 3, 3, 2, 5, 6, 1],
            ]
        )
        coefs = numpy.array([1, 2, 3, -5, 3, -2, -1, -2, 5, 2])

        def kernel(x):
            rmax = self.dist_sq(x, centers, e_mat, dist_type=1)
            return 1 / (1 + rmax)

        super(McCourt25, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.2, 0.6, 0.4, 0.8, 0.4, 0.3, 0.9, 0.8]
        self.fmin = -4.14042985928
        self.fmax = 5.47474174806
        self.classifiers = ["nonsmooth"]


class McCourt26(McCourtBase):
    def __init__(self, dim=3):
        assert dim == 3
        centers = numpy.array(
            [
                [0.5, 0.2, 0],
                [0.6, 0.2, 0.5],
                [0.4, 0.6, 0.5],
                [0.5, 0.7, 0.3],
                [0.4, 0.4, 0.4],
                [0.8, 0.5, 0.8],
                [0, 0, 0.8],
                [0.7, 0.7, 0.2],
                [0.9, 0.3, 1],
                [0.4, 0.4, 0.8],
                [0.2, 0.8, 0.8],
            ]
        )
        e_mat = 0.5 * numpy.array(
            [
                [2, 2, 2],
                [6, 5, 3],
                [3, 3, 3],
                [5, 2, 5],
                [4, 6, 3],
                [2, 2, 3],
                [2, 4, 1],
                [4, 6, 4],
                [1, 3, 4],
                [3, 2, 2],
                [6, 2, 3],
            ]
        )
        coefs = numpy.array([1, 2, 3, -5, 3, -2, 1, -2, 5, 2, -2])

        def kernel(x):
            rmax = self.dist_sq(x, centers, e_mat, dist_type=1)
            return numpy.exp(-rmax)

        super(McCourt26, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.5, 0.8, 0.3]
        self.fmin = -1.55349754312
        self.fmax = 5.97733366193
        self.classifiers = ["nonsmooth"]


class McCourt27(McCourtBase):
    def __init__(self, dim=3):
        assert dim == 3
        centers = numpy.array(
            [
                [0.6, 0.3, 0.5],
                [0.5, 0.2, 0],
                [0.4, 0.6, 0.5],
                [0.5, 0.7, 0.3],
                [0.4, 0.4, 0.4],
                [0.8, 0.5, 0.8],
                [0, 0, 0.8],
                [0.7, 0, 0.2],
                [0.9, 0.3, 1],
                [0.4, 0.4, 0.8],
                [0.2, 0.8, 0.8],
            ]
        )
        e_mat = 1 * numpy.array(
            [
                [2, 2, 2],
                [6, 5, 3],
                [3, 3, 3],
                [5, 2, 5],
                [4, 6, 3],
                [2, 2, 3],
                [2, 4, 1],
                [4, 6, 4],
                [1, 3, 4],
                [3, 2, 2],
                [6, 2, 3],
            ]
        )
        coefs = numpy.array([-10, 2, 3, 5, 3, 2, 1, 2, 5, 2, 2])

        def kernel(x):
            rmax = self.dist_sq(x, centers, e_mat, dist_type=1)
            return numpy.exp(-rmax)

        super(McCourt27, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.6, 0.3, 0.5]
        self.fmin = -1.76908456233
        self.fmax = 6.15634715165
        self.classifiers = ["nonsmooth", "unimodal"]


class McCourt28(McCourtBase):
    def __init__(self, dim=4):
        assert dim == 4
        centers = numpy.array(
            [
                [0.6, 0.2, 0.8, 0.4],
                [0.1, 0.1, 0.7, 0.9],
                [1, 0.1, 0.8, 0.6],
                [0, 0.3, 0.2, 1],
                [0.2, 1, 0.8, 0],
                [0.6, 0.9, 0.2, 0.9],
                [0.1, 0.7, 0.6, 0.8],
                [0.8, 0.4, 0.3, 0.2],
                [0.1, 1, 0.8, 0.2],
                [0.3, 0.9, 0.9, 0],
                [0.8, 1, 0.6, 0.9],
            ]
        )
        e_mat = 1 * numpy.array(
            [
                [1, 1, 1, 1],
                [5, 3, 3, 3],
                [4, 6, 2, 4],
                [4, 1, 6, 3],
                [2, 5, 3, 5],
                [5, 4, 6, 1],
                [6, 4, 1, 6],
                [5, 1, 2, 1],
                [1, 5, 4, 2],
                [1, 3, 3, 2],
                [4, 6, 6, 2],
            ]
        )
        coefs = numpy.array([-10, 2, 3, 5, 3, 2, 1, 2, 5, 2, 2])

        def kernel(x):
            r2 = self.dist_sq(x, centers, e_mat)
            return numpy.exp(-r2)

        super(McCourt28, self).__init__(dim, kernel, e_mat, coefs, centers)
        self.min_loc = [0.4493, 0.0667, 0.9083, 0.2710]
        self.fmin = -7.69432628909
        self.fmax = 9.13671993002
        self.classifiers = ["unimodal"]


class MegaDomain01(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(MegaDomain01, self).__init__(dim)
        self.bounds = [[0.1, 1], [1, 1000]]
        self.min_loc = [0.6, 200]
        self.fmin = 0.0
        self.fmax = 640000.0
        self.classifiers = ["unimodal", "unscaled"]

    def do_evaluate(self, x):
        return numpy.sum((x - self.min_loc) ** 2)


class MegaDomain02(TestFunction):
    def __init__(self, dim=3):
        assert dim == 3
        super(MegaDomain02, self).__init__(dim)
        self.bounds = [[0.0001, 0.1], [1, 10000], [40, 78901]]
        self.min_loc = [0.08, 2345, 12345]
        self.fmin = 0.0
        self.fmax = 4488300161.0
        self.classifiers = ["unimodal", "unscaled"]

    def do_evaluate(self, x):
        return numpy.sum((x - self.min_loc) ** 2)


class MegaDomain03(TestFunction):
    def __init__(self, dim=3):
        assert dim == 3
        super(MegaDomain03, self).__init__(dim)
        self.bounds = [[0.0001, 0.1], [1, 10000], [40, 78901]]
        self.min_loc = [0.08, 2345, 12345]
        self.fmin = -1.0
        self.fmax = 0.0
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        return -numpy.exp(
            -(numpy.sum((x - self.min_loc) / numpy.array([0.05, 6000, 34567])) ** 2)
        )


class MegaDomain04(TestFunction):
    def __init__(self, dim=3):
        assert dim == 3
        super(MegaDomain04, self).__init__(dim)
        self.bounds = [[0.0001, 0.1], [1, 10000], [40, 78901]]
        self.min_loc = [0.03, 1234, 65432]
        self.fmin = -1.1
        self.fmax = -0.04262395297
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        return -1.1 * numpy.exp(
            -abs(numpy.sum((x - self.min_loc) / numpy.array([0.05, 6000, 34567])))
        )


class MegaDomain05(TestFunction):
    def __init__(self, dim=4):
        assert dim == 4
        super(MegaDomain05, self).__init__(dim)
        self.bounds = [[0.0001, 0.1], [0.0001, 0.1], [1, 10000], [40, 78901]]
        self.min_loc = [0.0001, 0.04074477005, 1392.05038121473, 9185.44149117756]
        self.fmin = -1.0999
        self.fmax = 0.099999
        self.classifiers = ["bound_min"]

    def do_evaluate(self, x):
        exponent = numpy.sum(
            (x[1:] - numpy.array([0.02, 3333, 12345]))
            / numpy.array([0.05, 6000, 34567])
        )
        return x[0] - 1.1 * numpy.exp(-(exponent**2))


class Michalewicz(TestFunction):
    def __init__(self, dim=2):
        full_min_loc_vec = [
            2.202905513296628,
            1.570796322320509,
            1.284991564577549,
            1.923058467505610,
            1.720469766517768,
            1.570796319218113,
            1.454413962081172,
            1.756086513575824,
            1.655717409323190,
            1.570796319387859,
            1.497728796097675,
            1.923739461688219,
        ]
        full_fmin_vec = [
            0.8013034100985499,
            1,
            0.9590912698958649,
            0.9384624184720668,
            0.9888010806214966,
            1,
            0.9932271353558245,
            0.9828720362721659,
            0.9963943649250527,
            1,
            0.9973305415507061,
            0.9383447102236013,
        ]
        assert dim <= len(full_min_loc_vec)
        super(Michalewicz, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [pi] * self.dim)
        self.min_loc = full_min_loc_vec[:dim]
        self.fmin = -sum(full_fmin_vec[:dim])
        self.fmax = 0.0
        self.classifiers = ["boring", "complicated"]

    def do_evaluate(self, x):
        m = 10.0
        i = arange(1, self.dim + 1)
        return -sum(sin(x) * (sin(i * x**2 / pi)) ** (2 * m))


class MieleCantrell(TestFunction):
    def __init__(self, dim=4):
        assert dim == 4
        super(MieleCantrell, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.min_loc = [0, 1, 1, 1]
        self.fmin = 0
        self.fmax = 107.04280285028
        self.classifiers = ["boring", "bound_min"]

    def do_evaluate(self, x):
        x1, x2, x3, x4 = x
        return (exp(-x1) - x2) ** 4 + 100 * (x2 - x3) ** 6 + (tan(x3 - x4)) ** 4 + x1**8


class Mishra02(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Mishra02, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.min_loc = [1] * self.dim
        self.fmin = 2
        self.fmax = 9

    def do_evaluate(self, x):
        x1, x2 = x
        x_avg = self.dim - sum((x1 + x2) / 2)
        return (1 + x_avg) ** x_avg


class Mishra06(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Mishra06, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [2.88631, 1.82326]
        self.fmin = -2.28395
        self.fmax = 35.1518586485

    def do_evaluate(self, x):
        x1, x2 = x
        return -log(
            (
                (sin((cos(x1) + cos(x2)) ** 2) ** 2)
                - (cos((sin(x1) + sin(x2)) ** 2) ** 2)
                + x1
            )
            ** 2
        ) + 0.1 * ((x1 - 1) ** 2 + (x2 - 1) ** 2)


class Mishra08(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Mishra08, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [2, -3]
        self.fmin = 0
        self.fmax = 3.83363989364e18
        self.classifiers = ["unscaled", "boring"]

    def do_evaluate(self, x):
        x1, x2 = x
        f1 = abs(
            x1**10
            - 20 * x1**9
            + 180 * x1**8
            - 960 * x1**7
            + 3360 * x1**6
            - 8064 * x1**5
            + 13340 * x1**4
            - 15360 * x1**3
            + 11520 * x1**2
            - 5120 * x[0]
            + 2624
        )
        f2 = abs(x2**4 + 12 * x2**3 + 54 * x2**2 + 108 * x2 + 81)
        return 0.001 * (f1 + f2) ** 2


class Mishra10(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Mishra10, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [2, 2]
        self.fmin = 0
        self.fmax = 14400
        self.classifiers = ["discrete", "unscaled"]

    def do_evaluate(self, x):
        x1, x2 = int(x[0]), int(x[1])
        f1 = x1 + x2
        f2 = x1 * x2
        return (f1 - f2) ** 2


class ManifoldMin(TestFunction):
    def __init__(self, dim=2):
        super(ManifoldMin, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate([10] * self.dim)
        self.classifiers = ["nonsmooth", "multi_min", "unscaled"]

    def do_evaluate(self, x):
        return sum(abs(x)) * prod(abs(x))


class MixtureOfGaussians01(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(MixtureOfGaussians01, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [1] * self.dim)
        self.min_loc = [(-0.19870980807, -0.49764469526)]
        self.fmin = -0.50212488514
        self.fmax = -0.00001997307
        self.local_fmin = [-0.50212488514, -0.500001900968]
        self.classifiers = ["multimodal"]

    def do_evaluate(self, x):
        x1, x2 = x
        return -(
            0.5 * numpy.exp(-10 * (0.8 * (x1 + 0.2) ** 2 + 0.7 * (x2 + 0.5) ** 2))
            + 0.5 * numpy.exp(-8 * (0.3 * (x1 - 0.8) ** 2 + 0.6 * (x2 - 0.3) ** 2))
        )


class MixtureOfGaussians02(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(MixtureOfGaussians02, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [1] * self.dim)
        self.min_loc = [(-0.19945435737, -0.49900294852)]
        self.fmin = -0.70126732387
        self.fmax = -0.00001198419
        self.local_fmin = [-0.70126732387, -0.30000266214]
        self.classifiers = ["multimodal"]

    def do_evaluate(self, x):
        x1, x2 = x
        return -(
            0.7 * numpy.exp(-10 * (0.8 * (x1 + 0.2) ** 2 + 0.7 * (x2 + 0.5) ** 2))
            + 0.3 * numpy.exp(-8 * (0.3 * (x1 - 0.8) ** 2 + 0.6 * (x2 - 0.3) ** 2))
        )


class MixtureOfGaussians03(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(MixtureOfGaussians03, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [1] * self.dim)
        self.min_loc = [(-0.17918253215, -0.46292606370)]
        self.fmin = -0.63338923402
        self.fmax = -0.00993710053
        self.local_fmin = [-0.63338923402, -0.500001901929]
        self.classifiers = ["multimodal"]

    def do_evaluate(self, x):
        x1, x2 = x
        return -(
            0.5 * numpy.exp(-10 * (0.8 * (x1 + 0.2) ** 2 + 0.7 * (x2 + 0.5) ** 2))
            + 0.5 * numpy.exp(-2 * (0.3 * (x1 - 0.8) ** 2 + 0.6 * (x2 - 0.3) ** 2))
        )


class MixtureOfGaussians04(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(MixtureOfGaussians04, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [1] * self.dim)
        self.min_loc = [(-0.04454170197, 0.03290524075)]
        self.fmin = -0.582553299011
        self.fmax = -0.00207854059
        self.local_fmin = [
            -0.582553299011,
            -0.504982585841,
            -0.503213726167,
            -0.501693315297,
            -0.500412880827,
        ]
        self.classifiers = ["multimodal"]

    def do_evaluate(self, x):
        x1, x2 = x
        return -(
            0.5 * numpy.exp(-10 * (0.8 * (x1 + 0.8) ** 2 + 0.7 * (x2 + 0.8) ** 2))
            + 0.5 * numpy.exp(-8 * (0.3 * (x1 + 0.8) ** 2 + 0.6 * (x2 - 0.3) ** 2))
            + 0.5 * numpy.exp(-9 * (0.8 * x1**2 + 0.7 * x2**2))
            + 0.5 * numpy.exp(-9 * (0.8 * (x1 - 0.3) ** 2 + 0.7 * (x2 + 0.8) ** 2))
            + 0.5 * numpy.exp(-10 * (0.8 * (x1 - 0.8) ** 2 + 0.7 * (x2 - 0.8) ** 2))
        )


class MixtureOfGaussians05(TestFunction):
    def __init__(self, dim=8):
        assert dim == 8
        super(MixtureOfGaussians05, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [1] * self.dim)
        self.min_loc = [(-0.19870980798, -0.49764469559, 0, 0, 0, 0, 0, 0)]
        self.fmin = -0.50212691955
        self.fmax = -0.00001997307
        self.local_fmin = [-0.50212488514, -0.500001900968]
        self.classifiers = ["multimodal", "multi_min"]

    def do_evaluate(self, x):
        x1, x2, x3, x4, x5, x6, x7, x8 = x
        return -(
            0.5 * numpy.exp(-10 * (0.8 * (x1 + 0.2) ** 2 + 0.7 * (x2 + 0.5) ** 2))
            + 0.5 * numpy.exp(-8 * (0.3 * (x1 - 0.8) ** 2 + 0.6 * (x2 - 0.3) ** 2))
        )


class MixtureOfGaussians06(TestFunction):
    def __init__(self, dim=8):
        assert dim == 8
        super(MixtureOfGaussians06, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [1] * self.dim)
        self.min_loc = [(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)]
        self.fmin = -0.50016818373
        self.fmax = -0.00004539993
        self.classifiers = ["multi_min"]

    def do_evaluate(self, x):
        mu1 = 0.5 * numpy.ones(8)
        mu2 = -0.5 * numpy.ones(8)
        return -(
            0.5 * numpy.exp(-sum((x - mu1) ** 2))
            + 0.5 * numpy.exp(-sum((x - mu2) ** 2))
        )


class Ned01(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Ned01, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [-8.4666, -9.9988]
        self.fmin = -0.17894509347721144
        self.fmax = 1.18889613074
        self.classifiers = ["nonsmooth"]

    def do_evaluate(self, x):
        return abs(cos(sqrt(abs(x[0] ** 2 + x[1])))) ** 0.5 + 0.01 * x[0] + 0.01 * x[1]


class Ned03(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Ned03, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [-1.98682, -10]
        self.fmin = -1.019829
        self.fmax = 144.506592895
        self.classifiers = ["bound_min"]

    def do_evaluate(self, x):
        x1, x2 = x
        f1 = sin((cos(x1) + cos(x2)) ** 2) ** 2
        f2 = cos((sin(x1) + sin(x2)) ** 2) ** 2
        return (f1 + f2 + x1) ** 2 + 0.01 * x1 + 0.1 * x2


class OddSquare(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(OddSquare, self).__init__(dim)
        self.bounds = lzip([-3 * pi] * self.dim, [3 * pi] * self.dim)
        self.min_loc = [0.912667308214834, 1.212667322565022]
        self.fmin = -1.008467279147474
        self.fmax = 0.870736981456
        self.classifiers = ["boring"]

    def do_evaluate(self, x):
        b = asarray([1, 1.3])
        d = self.dim * max((x - b) ** 2)
        h = sum((x - b) ** 2)
        return -exp(-d / (2 * pi)) * cos(pi * d) * (1 + 0.02 * h / (d + 0.01))


class Parsopoulos(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Parsopoulos, self).__init__(dim)
        self.bounds = lzip([-5] * self.dim, [5] * self.dim)
        self.min_loc = [pi / 2, pi]
        self.fmin = 0
        self.fmax = 2
        self.classifiers = ["oscillatory", "multi_min"]

    def do_evaluate(self, x):
        x1, x2 = x
        return cos(x1) ** 2 + sin(x2) ** 2


class Pavianini(TestFunction):
    def __init__(self, dim=10):
        assert dim == 10
        super(Pavianini, self).__init__(dim)
        self.bounds = lzip([2.001] * self.dim, [9.999] * self.dim)
        self.min_loc = [9.350266] * self.dim
        self.fmin = -45.7784684040686
        self.fmax = 516.402401423

    def do_evaluate(self, x):
        return sum(log(x - 2) ** 2 + log(10 - x) ** 2) - prod(x) ** 0.2


class Penalty01(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Penalty01, self).__init__(dim)
        self.bounds = lzip([-4] * self.dim, [4] * self.dim)
        self.min_loc = [-1] * self.dim
        self.fmin = 0
        self.fmax = 2.34982038483

    def do_evaluate(self, x):
        y1, y2 = 1 + (x + 1) / 4
        return (pi / 30) * (
            10 * sin(pi * y1) ** 2
            + (y1 - 1) ** 2 * (1 + 10 * sin(pi * y2) ** 2)
            + (y2 - 1) ** 2
        )


class Penalty02(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Penalty02, self).__init__(dim)
        self.bounds = lzip([-4] * self.dim, [4] * self.dim)
        self.min_loc = [1] * self.dim
        self.fmin = 0
        self.fmax = 9.10735658210
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        x1, x2 = x
        return 0.1 * (
            10 * sin(3 * pi * x1) ** 2
            + (x1 - 1) ** 2 * (1 + sin(pi * x2) ** 2)
            + (x2 - 1) ** 2 * (1 + sin(2 * pi * x2) ** 2)
        )


class PenHolder(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(PenHolder, self).__init__(dim)
        self.bounds = lzip([-11] * self.dim, [11] * self.dim)
        self.min_loc = [-9.646167708023526, 9.646167671043401]
        self.fmin = -0.9635348327265058
        self.fmax = 0
        self.classifiers = ["nonsmooth"]

    def do_evaluate(self, x):
        x1, x2 = x
        return -exp(
            -1 / (abs(cos(x1) * cos(x2) * exp(abs(1 - sqrt(x1**2 + x2**2) / pi))))
        )


class Perm01(TestFunction):
    def __init__(self, dim=2):
        assert dim > 1
        super(Perm01, self).__init__(dim)
        self.bounds = lzip([-self.dim] * self.dim, [self.dim + 1] * self.dim)
        self.min_loc = [1] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate([self.dim + 1] * self.dim)
        self.classifiers = ["unscaled"]

    def do_evaluate(self, x):
        return sum(
            sum(
                [
                    (j**k + 0.5) * ((x[j - 1] / j) ** k - 1)
                    for j in range(1, self.dim + 1)
                ]
            )
            ** 2
            for k in range(1, self.dim + 1)
        )


class Perm02(TestFunction):
    def __init__(self, dim=2):
        assert dim > 1
        super(Perm02, self).__init__(dim)
        self.bounds = lzip([-self.dim] * self.dim, [self.dim + 1] * self.dim)
        self.min_loc = 1 / arange(1, self.dim + 1)
        self.fmin = 0
        self.fmax = self.do_evaluate([self.dim + 1] * self.dim)
        self.classifiers = ["unscaled"]

    def do_evaluate(self, x):
        return sum(
            sum(
                [
                    (j + 10) * (x[j - 1] ** k - (1.0 / j) ** k)
                    for j in range(1, self.dim + 1)
                ]
            )
            ** 2
            for k in range(1, self.dim + 1)
        )


class Pinter(TestFunction):
    def __init__(self, dim=2):
        assert dim > 1
        super(Pinter, self).__init__(dim)
        self.bounds = lzip([-5] * self.dim, [2] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate([-5] * self.dim)

    def do_evaluate(self, x):
        f = 0
        for i in range(self.dim):
            x_i = x[i]
            if i == 0:
                x_mi = x[-1]
                x_pi = x[i + 1]
            elif i == self.dim - 1:
                x_mi = x[i - 1]
                x_pi = x[0]
            else:
                x_mi = x[i - 1]
                x_pi = x[i + 1]
            a = x_mi * sin(x_i) + sin(x_pi)
            b = x_mi**2 - 2 * x_i + 3 * x_pi - cos(x_i) + 1
            f += (
                (i + 1) * x_i**2
                + 20 * (i + 1) * sin(a) ** 2
                + (i + 1) * log10(1 + (i + 1) * b**2)
            )
        return f


class Plateau(TestFunction):
    def __init__(self, dim=2):
        super(Plateau, self).__init__(dim)
        self.bounds = lzip([-2.34] * self.dim, [5.12] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 30
        self.fmax = self.do_evaluate([5.12] * self.dim)
        self.classifiers = ["discrete", "unimodal"]

    def do_evaluate(self, x):
        return 30 + sum(floor(abs(x)))


class Powell(TestFunction):
    def __init__(self, dim=4):
        assert dim == 4
        super(Powell, self).__init__(dim)
        self.bounds = lzip([-4] * self.dim, [5] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 105962
        self.classifiers = ["unscaled"]

    def do_evaluate(self, x):
        x1, x2, x3, x4 = x
        return (
            (x1 + 10 * x2) ** 2
            + 5 * (x3 - x4) ** 2
            + (x2 - 2 * x3) ** 4
            + 10 * (x1 - x4) ** 4
        )


class PowellTripleLog(TestFunction):
    def __init__(self, dim=12):
        assert dim == 12
        super(PowellTripleLog, self).__init__(dim)
        self.bounds = lzip([-4] * self.dim, [1] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 10.46587093572

    def do_evaluate(self, x):
        return log(
            1
            + sum(
                [
                    Powell().do_evaluate(x_subset)
                    for x_subset in (x[0:4], x[4:8], x[8:12])
                ]
            )
        )


class PowerSum(TestFunction):
    def __init__(self, dim=4):
        assert dim == 4
        super(PowerSum, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [4] * self.dim)
        self.min_loc = [1, 2, 2, 3]
        self.fmin = 0
        self.fmax = 875224
        self.classifiers = ["unscaled", "multi_min"]

    def do_evaluate(self, x):
        b = [8, 18, 44, 114]
        return sum(
            [(sum([xx ** (k + 1) for xx in x]) - bb) ** 2 for k, bb in enumerate(b)]
        )


class Price(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Price, self).__init__(dim)
        self.bounds = lzip([-15] * self.dim, [15] * self.dim)
        self.min_loc = [5, 5]
        self.fmin = 0
        self.fmax = self.do_evaluate(asarray([15] * self.dim))
        self.classifiers = ["multi_min", "nonsmooth"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (abs(x1) - 5) ** 2 + (abs(x2) - 5) ** 2


class Qing(TestFunction):
    def __init__(self, dim=2):
        assert dim < 100  # If greater, the optimum is on the boundary
        super(Qing, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [sqrt(x) for x in range(1, self.dim + 1)]
        self.fmin = 0
        self.fmax = self.do_evaluate(numpy.max(self.bounds, axis=1))
        self.classifiers = ["multi_min"]

    def do_evaluate(self, x):
        return sum((x**2 - numpy.arange(1, self.dim + 1)) ** 2)


class Quadratic(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Quadratic, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [0.19388, 0.48513]
        self.fmin = -3873.72418
        self.fmax = 51303.16
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            -3803.84
            - 138.08 * x1
            - 232.92 * x2
            + 128.08 * x1**2
            + 203.64 * x2**2
            + 182.25 * x1 * x2
        )


class Rastrigin(TestFunction):
    def __init__(self, dim=8):
        assert dim == 8
        super(Rastrigin, self).__init__(dim)
        self.bounds = lzip([-5, -5, -2, -2, -5, -5, -2, -2], [2, 2, 5, 5, 2, 2, 5, 5])
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 280.61197450173

    def do_evaluate(self, x):
        return 10 * self.dim + sum(x**2 - 10 * cos(2 * pi * x))


class RippleSmall(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(RippleSmall, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.min_loc = [0.1] * self.dim
        self.fmin = -2.2
        self.fmax = 0
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        return sum(
            -exp(-2 * log(2) * ((x - 0.1) / 0.8) ** 2)
            * (sin(5 * pi * x) ** 6 + 0.1 * cos(500 * pi * x) ** 2)
        )


class RippleBig(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(RippleBig, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.min_loc = [0.1] * self.dim
        self.fmin = -2
        self.fmax = 0
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        return sum(-exp(-2 * log(2) * ((x - 0.1) / 0.8) ** 2) * (sin(5 * pi * x) ** 6))


class RosenbrockLog(TestFunction):
    def __init__(self, dim=11):
        assert dim == 11
        super(RosenbrockLog, self).__init__(dim)
        self.bounds = [
            [-2, 2],
            [-2, 1.1],
            [0.5, 2],
            [-2, 2],
            [0.8, 2],
            [-2, 1.5],
            [-2, 2],
            [-2, 1.2],
            [0.7, 2],
            [-2, 2],
            [-2, 2],
        ]
        self.min_loc = [1] * self.dim
        self.fmin = 0
        self.fmax = 10.09400460102

    def do_evaluate(self, x):
        return log(1 + sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


class RosenbrockModified(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(RosenbrockModified, self).__init__(dim)
        self.bounds = lzip([-2] * self.dim, [2] * self.dim)
        self.min_loc = [-0.909553754255364, -0.950571727005927]
        self.fmin = 34.040243106640787
        self.fmax = 3682.99999918

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            74
            + 100 * (x2 - x1**2) ** 2
            + (1 - x1) ** 2
            - 400 * exp(-((x1 + 1) ** 2 + (x2 + 1) ** 2) / 0.1)
        )


class Salomon(TestFunction):
    def __init__(self, dim=2):
        super(Salomon, self).__init__(dim)
        self.bounds = lzip([-100] * self.dim, [50] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(asarray([-100] * self.dim))

    def do_evaluate(self, x):
        return 1 - cos(2 * pi * sqrt(sum(x**2))) + 0.1 * sqrt(sum(x**2))


class Sargan(TestFunction):
    def __init__(self, dim=2):
        assert dim > 1
        super(Sargan, self).__init__(dim)
        self.bounds = lzip([-2] * self.dim, [4] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(asarray([4] * self.dim))
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        x0 = x[:-1]
        x1 = roll(x, -1)[:-1]
        return sum(self.dim * (x**2 + 0.4 * sum(x0 * x1)))


class Schaffer(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Schaffer, self).__init__(dim)
        self.bounds = [[-10, 30], [-30, 10]]
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 0.997860938826
        self.classifiers = ["boring"]

    def do_evaluate(self, x):
        x1, x2 = x
        return 0.5 + (sin((x1**2 + x2**2) ** 2) ** 2 - 0.5) / (
            1 + 0.001 * (x1**2 + x2**2) ** 2
        )


class SchmidtVetters(TestFunction):
    def __init__(self, dim=3):
        assert dim == 3
        super(SchmidtVetters, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [10] * self.dim)
        self.min_loc = [3.79367424567, 3.79367424352, 3.78978412518]
        self.fmin = 3
        self.fmax = -0.99009900990
        self.classifiers = ["oscillatory", "multi_min"]

    def do_evaluate(self, x):
        x1, x2, x3 = x
        return (
            1 / (1 + (x1 - x2) ** 2)
            + sin(0.5 * (pi * x2 + x3))
            + exp(-(((x1 + x2) / (x2 + 1e-16) - 2) ** 2))
        )


class Schwefel01(TestFunction):
    def __init__(self, dim=2):
        super(Schwefel01, self).__init__(dim)
        self.bounds = lzip([-100] * self.dim, [20] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(asarray([-100] * self.dim))
        self.classifiers = ["unscaled", "unimodal"]

    def do_evaluate(self, x):
        return (sum(x**2)) ** sqrt(pi)


class Schwefel06(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Schwefel06, self).__init__(dim)
        self.bounds = lzip([-50] * self.dim, [100] * self.dim)
        self.min_loc = [1, 3]
        self.fmin = 0
        self.fmax = 295
        self.classifiers = ["unimodal", "nonsmooth"]

    def do_evaluate(self, x):
        x1, x2 = x
        return max([abs(x1 + 2 * x2 - 7), abs(2 * x1 + x2 - 5)])


class Schwefel20(TestFunction):
    def __init__(self, dim=2):
        super(Schwefel20, self).__init__(dim)
        self.bounds = lzip([-60] * self.dim, [100] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(asarray([100] * self.dim))
        self.classifiers = ["unimodal", "nonsmooth"]

    def do_evaluate(self, x):
        return sum(abs(x))


class Schwefel22(TestFunction):
    def __init__(self, dim=2):
        super(Schwefel22, self).__init__(dim)
        self.bounds = lzip([-5] * self.dim, [10] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(asarray([10] * self.dim))
        self.classifiers = ["unimodal", "nonsmooth"]

    def do_evaluate(self, x):
        return sum(abs(x)) + prod(abs(x))


class Schwefel26(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Schwefel26, self).__init__(dim)
        self.bounds = lzip([-500] * self.dim, [500] * self.dim)
        self.min_loc = [420.968746] * self.dim
        self.fmin = 0
        self.fmax = 1675.92130876
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        return 418.982887 * self.dim - sum([x * sin(sqrt(abs(x)))])


class Schwefel36(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Schwefel36, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [20] * self.dim)
        self.min_loc = [12, 12]
        self.fmin = -3456
        self.fmax = 3200
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        x1, x2 = x
        return -x1 * x2 * (72 - 2 * x1 - 2 * x2)


class Shekel05(TestFunction):
    def __init__(self, dim=4):
        assert dim == 4
        super(Shekel05, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [10] * self.dim)
        self.min_loc = [4] * self.dim
        self.fmin = -10.152719932456289
        self.fmax = -0.0377034398748
        self.classifiers = ["boring"]

    def do_evaluate(self, x):
        a_mat = asarray(
            [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7]]
        )
        c_vec = asarray([0.1, 0.2, 0.2, 0.4, 0.6])
        return -sum(1 / (dot(x - a, x - a) + c) for a, c in lzip(a_mat, c_vec))


class Shekel07(TestFunction):
    def __init__(self, dim=4):
        assert dim == 4
        super(Shekel07, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [10] * self.dim)
        self.min_loc = [4] * self.dim
        self.fmin = -10.3999
        self.fmax = -0.0503833861496
        self.classifiers = ["boring"]

    def do_evaluate(self, x):
        a_mat = asarray(
            [
                [4, 4, 4, 4],
                [1, 1, 1, 1],
                [8, 8, 8, 8],
                [6, 6, 6, 6],
                [3, 7, 3, 7],
                [2, 9, 2, 9],
                [5, 5, 3, 3],
            ]
        )
        c_vec = asarray([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3])
        return -sum(1 / (dot(x - a, x - a) + c) for a, c in lzip(a_mat, c_vec))


class Shekel10(TestFunction):
    def __init__(self, dim=4):
        assert dim == 4
        super(Shekel10, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [10] * self.dim)
        self.min_loc = [4] * self.dim
        self.fmin = -10.5319
        self.fmax = -0.0784208993809
        self.classifiers = ["boring"]

    def do_evaluate(self, x):
        a_mat = asarray(
            [
                [4, 4, 4, 4],
                [1, 1, 1, 1],
                [8, 8, 8, 8],
                [6, 6, 6, 6],
                [3, 7, 3, 7],
                [2, 9, 2, 9],
                [5, 5, 3, 3],
                [8, 1, 8, 1],
                [6, 2, 6, 2],
                [7, 3, 7, 3],
            ]
        )
        c_vec = asarray([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        return -sum(1 / (dot(x - a, x - a) + c) for a, c in lzip(a_mat, c_vec))


class Shubert01(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Shubert01, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [-7.0835, 4.8580]
        self.fmin = -186.7309
        self.fmax = 210.448484805
        self.classifiers = ["multi_min", "oscillatory"]

    def do_evaluate(self, x):
        return prod([sum([i * cos((i + 1) * xx + i) for i in range(1, 6)]) for xx in x])


class Shubert03(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Shubert03, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [5.791794, 5.791794]
        self.fmin = -24.062499
        self.fmax = 29.675796163
        self.classifiers = ["multi_min", "oscillatory"]

    def do_evaluate(self, x):
        return (
            -sin(2 * x[0] + 1)
            - 2 * sin(3 * x[0] + 2)
            - 3 * sin(4 * x[0] + 3)
            - 4 * sin(5 * x[0] + 4)
            - 5 * sin(6 * x[0] + 5)
            - sin(2 * x[1] + 1)
            - 2 * sin(3 * x[1] + 2)
            - 3 * sin(4 * x[1] + 3)
            - 4 * sin(5 * x[1] + 4)
            - 5 * sin(6 * x[1] + 5)
        )


class SineEnvelope(TestFunction):
    def __init__(self, dim=2):
        assert dim > 1
        super(SineEnvelope, self).__init__(dim)
        self.bounds = lzip([-20] * self.dim, [10] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.dim - 1
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        x_sq = x[0:-1] ** 2 + x[1:] ** 2
        return sum((sin(sqrt(x_sq)) ** 2 - 0.5) / (1 + 0.001 * x_sq) ** 2 + 0.5)


class SixHumpCamel(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(SixHumpCamel, self).__init__(dim)
        self.bounds = [[-2, 2], [-1.5, 1.5]]
        self.min_loc = [0.08984201368301331, -0.7126564032704135]
        self.fmin = -1.031628
        self.fmax = 17.98333333333
        self.classifiers = ["multi_min"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (4 * x2**2 - 4) * x2**2


class Sphere(TestFunction):
    def __init__(self, dim=2):
        super(Sphere, self).__init__(dim)
        self.bounds = lzip([-5.12] * self.dim, [2.12] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(asarray([-5.12] * self.dim))
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        return sum(x**2)


class Step(TestFunction):
    def __init__(self, dim=2):
        super(Step, self).__init__(dim)
        self.bounds = lzip([-5] * self.dim, [5] * self.dim)
        self.min_loc = [0.5] * self.dim
        self.fmin = self.do_evaluate(asarray([0] * self.dim))
        self.fmax = self.do_evaluate(asarray([5] * self.dim))
        self.classifiers = ["discrete", "unimodal"]

    def do_evaluate(self, x):
        return sum((floor(x) + 0.5) ** 2)


class StretchedV(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(StretchedV, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [5] * self.dim)
        self.min_loc = [-9.38723188, 9.34026753]
        self.fmin = 0
        self.fmax = 3.47171564062
        self.classifiers = ["oscillatory", "multi_min"]

    def do_evaluate(self, x):
        r = sum(x**2)
        return r**0.25 * (sin(50 * r**0.1 + 1)) ** 2


class StyblinskiTang(TestFunction):
    def __init__(self, dim=2):
        super(StyblinskiTang, self).__init__(dim)
        self.bounds = lzip([-5] * self.dim, [5] * self.dim)
        self.min_loc = [-2.903534018185960] * self.dim
        self.fmin = -39.16616570377142 * self.dim
        self.fmax = self.do_evaluate(asarray([5] * self.dim))

    def do_evaluate(self, x):
        return sum(x**4 - 16 * x**2 + 5 * x) / 2


class SumPowers(TestFunction):
    def __init__(self, dim=2):
        super(SumPowers, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [0.5] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(asarray([-1] * self.dim))
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        return sum([abs(x) ** (i + 1) for i in range(1, self.dim + 1)])


class TestTubeHolder(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(TestTubeHolder, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [-pi / 2, 0]
        self.fmin = -10.87229990155800
        self.fmax = 0
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        x1, x2 = x
        return -4 * abs(sin(x1) * cos(x2) * exp(abs(cos((x1**2 + x2**2) / 200))))


class ThreeHumpCamel(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(ThreeHumpCamel, self).__init__(dim)
        self.bounds = lzip([-5] * self.dim, [5] * self.dim)
        self.min_loc = [0, 0]
        self.fmin = 0
        self.fmax = 2047.91666667

    def do_evaluate(self, x):
        x1, x2 = x
        return 2 * x1**2 - 1.05 * x1**4 + x1**6 / 6 + x1 * x2 + x2**2


class Trefethen(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Trefethen, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [-0.02440307923, 0.2106124261]
        self.fmin = -3.3068686474
        self.fmax = 56.1190428617
        self.classifiers = ["complicated"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            exp(sin(50 * x1))
            + sin(60 * exp(x2))
            + sin(70 * sin(x1))
            + sin(sin(80 * x2))
            - sin(10 * (x1 + x2))
            + 0.25 * (x1**2 + x2**2)
        )


class Trid(TestFunction):
    def __init__(self, dim=6):
        assert dim == 6
        super(Trid, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [20] * self.dim)
        self.min_loc = [6, 10, 12, 12, 10, 6]
        self.fmin = -50
        self.fmax = 1086

    def do_evaluate(self, x):
        return sum((x - 1) ** 2) - sum(x[1:] * x[0:-1])


class Tripod(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Tripod, self).__init__(dim)
        self.bounds = lzip([-100] * self.dim, [100] * self.dim)
        self.min_loc = [0, -50]
        self.fmin = 0
        self.fmax = 150
        self.classifiers = ["nonsmooth"]

    def do_evaluate(self, x):
        x1, x2 = x
        p1 = float(x1 >= 0)
        p2 = float(x2 >= 0)
        return (
            p2 * (1 + p1)
            + abs(x1 + 50 * p2 * (1 - 2 * p1))
            + abs(x2 + 50 * (1 - 2 * p2))
        )


class Ursem01(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Ursem01, self).__init__(dim)
        self.bounds = [(-2.5, 3), (-2, 2)]
        self.min_loc = [1.69714, 0]
        self.fmin = -4.8168
        self.fmax = 2.7821026951

    def do_evaluate(self, x):
        x1, x2 = x
        return -sin(2 * x1 - 0.5 * pi) - 3 * cos(x2) - 0.5 * x1


class Ursem03(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Ursem03, self).__init__(dim)
        self.bounds = [[-2, 1], [-1.5, 1.5]]
        self.min_loc = [0] * self.dim
        self.fmin = -3
        self.fmax = 1.98893400593
        self.classifiers = ["nonsmooth", "oscillatory"]

    def do_evaluate(self, x):
        x1, x2 = x
        return -sin(2.2 * pi * x1 + 0.5 * pi) * ((2 - abs(x1)) / 2) * (
            (3 - abs(x1)) / 2
        ) - sin(2.2 * pi * x2 + 0.5 * pi) * ((2 - abs(x2)) / 2) * ((3 - abs(x2)) / 2)


class Ursem04(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Ursem04, self).__init__(dim)
        self.bounds = lzip([-2] * self.dim, [1.5] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = -1.5
        self.fmax = 0.267902882972
        self.classifiers = ["nonsmooth", "unimodal"]

    def do_evaluate(self, x):
        x1, x2 = x
        return -3 * sin(0.5 * pi * x1 + 0.5 * pi) * (2 - sqrt(x1**2 + x2**2)) / 4


class UrsemWaves(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(UrsemWaves, self).__init__(dim)
        self.bounds = [(-0.9, 1.2), (-1.2, 1.2)]
        self.min_loc = [1.2] * self.dim
        self.fmin = -8.5536
        self.fmax = 7.71938723147
        self.classifiers = ["bound_min"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            -0.9 * x1**2
            + (x2**2 - 4.5 * x2**2) * x1 * x2
            + 4.7 * cos(3 * x1 - x2**2 * (2 + x1)) * sin(2.5 * pi * x1)
        )


class VenterSobiezcczanskiSobieski(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(VenterSobiezcczanskiSobieski, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [5] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = -400
        self.fmax = 4920.34496357
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            x1**2
            - 100 * cos(x1) ** 2
            - 100 * cos(x1**2 / 30)
            + x2**2
            - 100 * cos(x2) ** 2
            - 100 * cos(x2**2 / 30)
        )


class Watson(TestFunction):
    def __init__(self, dim=6):
        assert dim == 6
        super(Watson, self).__init__(dim)
        self.bounds = lzip([-2] * self.dim, [2] * self.dim)
        self.min_loc = [-0.0158, 1.012, -0.2329, 1.260, -1.513, 0.9928]
        self.fmin = 0.002288
        self.fmax = 3506782.05596
        self.classifiers = ["unscaled"]

    def do_evaluate(self, x):
        vec = zeros((31,))
        div = (arange(29) + 1) / 29
        s1 = 0
        dx = 1
        for j in range(1, self.dim):
            s1 += j * dx * x[j]
            dx *= div
        s2 = 0
        dx = 1
        for j in range(self.dim):
            s2 += dx * x[j]
            dx *= div
        vec[:29] = s1 - s2**2 - 1
        vec[29] = x[0]
        vec[30] = x[1] - x[0] ** 2 - 1
        return sum(vec**2)


class Weierstrass(TestFunction):
    def __init__(self, dim=2):
        super(Weierstrass, self).__init__(dim)
        self.bounds = lzip([-0.5] * self.dim, [0.2] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = self.do_evaluate(asarray(self.min_loc))
        self.fmax = self.do_evaluate(asarray([-0.5] * self.dim))
        self.classifiers = ["complicated"]

    def do_evaluate(self, x):
        a, b, kmax = 0.5, 3, 20
        ak = a ** (numpy.arange(0, kmax + 1))
        bk = b ** (numpy.arange(0, kmax + 1))
        return sum(
            [
                sum(ak * cos(2 * pi * bk * (xx + 0.5)))
                - self.dim * sum(ak * cos(pi * bk))
                for xx in x
            ]
        )


class Wolfe(TestFunction):
    def __init__(self, dim=3):
        assert dim == 3
        super(Wolfe, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [2] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = -1
        self.fmax = 6.30351707066

    def do_evaluate(self, x):
        x1, x2, x3 = x
        return (4 / 3) * (x1**2 + x2**2 - x1 * x2) ** 0.75 + x3


class XinSheYang02(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(XinSheYang02, self).__init__(dim)
        self.bounds = lzip([-pi] * self.dim, [2 * pi] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 88.8266046808
        self.classifiers = ["nonsmooth", "unscaled"]

    def do_evaluate(self, x):
        return sum(abs(x)) * exp(-sum(sin(x**2)))


class XinSheYang03(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(XinSheYang03, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [20] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = -1
        self.fmax = 1
        self.classifiers = ["boring", "unimodal"]

    def do_evaluate(self, x):
        beta, m = 15, 5
        return exp(-sum((x / beta) ** (2 * m))) - 2 * exp(-sum(x**2)) * prod(
            cos(x) ** 2
        )


class Xor(TestFunction):
    def __init__(self, dim=9):
        assert dim == 9
        super(Xor, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [1] * self.dim)
        self.min_loc = [1, -1, 1, -1, -1, 1, 1, -1, 0.421457080713797]
        self.fmin = 0.959758757011962
        self.fmax = 1.77818910738
        self.classifiers = ["bound_min"]

    def do_evaluate(self, x):
        f11 = x[6] / (1 + exp(-x[0] - x[1] - x[4]))
        f12 = x[7] / (1 + exp(-x[2] - x[3] - x[5]))
        f1 = (1 + exp(-f11 - f12 - x[8])) ** (-2)
        f21 = x[6] / (1 + exp(-x[4]))
        f22 = x[7] / (1 + exp(-x[5]))
        f2 = (1 + exp(-f21 - f22 - x[8])) ** (-2)
        f31 = x[6] / (1 + exp(-x[0] - x[4]))
        f32 = x[7] / (1 + exp(-x[2] - x[5]))
        f3 = (1 - (1 + exp(-f31 - f32 - x[8])) ** (-1)) ** 2
        f41 = x[6] / (1 + exp(-x[1] - x[4]))
        f42 = x[7] / (1 + exp(-x[3] - x[5]))
        f4 = (1 - (1 + exp(-f41 - f42 - x[8])) ** (-1)) ** 2
        return f1 + f2 + f3 + f4


class YaoLiu(TestFunction):
    def __init__(self, dim=2):
        super(YaoLiu, self).__init__(dim)
        self.bounds = lzip([-5.12] * self.dim, [2] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(asarray([-4.52299366685] * self.dim))
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        return sum(x**2 - 10 * cos(2 * pi * x) + 10)


class ZeroSum(TestFunction):
    def __init__(self, dim=2):
        super(ZeroSum, self).__init__(dim)
        self.bounds = lzip([-8] * self.dim, [6] * self.dim)
        self.min_loc = [0] * self.dim
        self.fmin = 1
        self.fmax = self.do_evaluate(asarray([-8] * self.dim))
        self.classifiers = ["nonsmooth", "multi_min"]

    def do_evaluate(self, x):
        return 1 + (10000 * abs(sum(x))) ** 0.5


class Zimmerman(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Zimmerman, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [8] * self.dim)
        self.min_loc = [7, 2]
        self.fmin = 0
        self.fmax = 3000
        self.classifiers = ["nonsmooth", "multi_min"]

    def do_evaluate(self, x):
        zh1 = lambda v: 9 - v[0] - v[1]
        zh2 = lambda v: (v[0] - 3) ** 2 + (v[1] - 2) ** 2 - 16
        zh3 = lambda v: v[0] * v[1] - 14
        zp = lambda v: 100 * (1 + v)
        px = [
            zh1(x),
            zp(zh2(x)) * sign(zh2(x)),
            zp(zh3(x)) * sign(zh3(x)),
            zp(-x[0]) * sign(x[0]),
            zp(-x[1]) * sign(x[1]),
        ]
        return numpy.fmin(max(px), self.fmax)


# Below are all 1D functions
class Problem02(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem02, self).__init__(dim)
        self.bounds = [(2.7, 7.5)]
        self.min_loc = 5.145735285687302
        self.fmin = -1.899599349152113
        self.fmax = 0.888314780101

    def do_evaluate(self, x):
        x = x[0]
        return sin(x) + sin(3.33333 * x)


class Problem03(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem03, self).__init__(dim)
        self.bounds = [(-10, 10)]
        self.min_loc = -6.7745761
        self.fmin = -12.03124
        self.fmax = 14.8379500232
        self.classifiers = ["oscillatory", "multi_min"]

    def do_evaluate(self, x):
        x = x[0]
        return -sum([k * sin((k + 1) * x + k) for k in range(1, 6)])


class Problem04(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem04, self).__init__(dim)
        self.bounds = [(1.9, 3.9)]
        self.min_loc = 7 / 4 + numpy.sqrt(5) / 2
        self.fmin = -3.850450708800220
        self.fmax = -2.56659750586
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        x = x[0]
        return -(16 * x**2 - 24 * x + 5) * exp(-x)


class Problem05(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem05, self).__init__(dim)
        self.bounds = [(0, 1.2)]
        self.min_loc = 0.96609
        self.fmin = -1.48907
        self.fmax = 2.01028135138

    def do_evaluate(self, x):
        x = x[0]
        return -(1.4 - 3 * x) * sin(18 * x)


class Problem06(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem06, self).__init__(dim)
        self.bounds = [(-10, 10)]
        self.min_loc = 0.67956
        self.fmin = -0.824239
        self.fmax = 0.824239398459
        self.classifiers = ["unimodal", "boring"]

    def do_evaluate(self, x):
        x = x[0]
        return -(x + sin(x)) * exp(-(x**2))


class Problem07(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem07, self).__init__(dim)
        self.bounds = [(2.7, 7.5)]
        self.min_loc = 5.199778369093496
        self.fmin = -1.601307546494395
        self.fmax = 2.56475013849

    def do_evaluate(self, x):
        x = x[0]
        return sin(x) + sin(10 / 3 * x) + log(x) - 0.84 * x + 3


class Problem09(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem09, self).__init__(dim)
        self.bounds = [(3.1, 20.4)]
        self.min_loc = 17.039
        self.fmin = -1.90596
        self.fmax = 1.858954715

    def do_evaluate(self, x):
        x = x[0]
        return sin(x) + sin(2 / 3 * x)


class Problem10(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem10, self).__init__(dim)
        self.bounds = [(0, 10)]
        self.min_loc = 7.9787
        self.fmin = -7.916727
        self.fmax = 5.44021110889

    def do_evaluate(self, x):
        x = x[0]
        return -x * sin(x)


class Problem11(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem11, self).__init__(dim)
        self.bounds = [(-pi / 2, 2 * pi)]
        self.min_loc = 2.09439
        self.fmin = -1.5
        self.fmax = 3
        self.classifiers = ["multi_min"]

    def do_evaluate(self, x):
        x = x[0]
        return 2 * cos(x) + cos(2 * x)


class Problem12(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem12, self).__init__(dim)
        self.bounds = [(0, 2 * pi)]
        self.min_loc = pi
        self.fmin = -1
        self.fmax = 1
        self.classifiers = ["multi_min"]

    def do_evaluate(self, x):
        x = x[0]
        return (sin(x)) ** 3 + (cos(x)) ** 3


class Problem13(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem13, self).__init__(dim)
        self.bounds = [(0.001, 0.99)]
        self.min_loc = 1 / sqrt(2)
        self.fmin = -1.587401051968199
        self.fmax = -1.00999966667
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        x = x[0]
        return -(x**0.66666) - (1 - x**2) ** 0.33333


class Problem14(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem14, self).__init__(dim)
        self.bounds = [(0, 4)]
        self.min_loc = 0.224885
        self.fmin = -0.788685
        self.fmax = 0.47836186833
        self.classifiers = ["oscillatory"]

    def do_evaluate(self, x):
        x = x[0]
        return -exp(-x) * sin(2 * pi * x)


class Problem15(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem15, self).__init__(dim)
        self.bounds = [(-5, 5)]
        self.min_loc = 2.414194788875151
        self.fmin = -0.035533905879289
        self.fmax = 7.03553390593
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        x = x[0]
        return (x**2 - 5 * x + 6) / (x**2 + 1)


class Problem18(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem18, self).__init__(dim)
        self.bounds = [(0, 6)]
        self.min_loc = 2
        self.fmin = 0
        self.fmax = 4
        self.classifiers = ["unimodal"]

    def do_evaluate(self, x):
        x = x[0]
        if x <= 3:
            return (x - 2) ** 2
        return 2 * log(x - 2) + 1


class Problem20(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem20, self).__init__(dim)
        self.bounds = [(-10, 10)]
        self.min_loc = 1.195137
        self.fmin = -0.0634905
        self.fmax = 0.0634905289316
        self.classifiers = ["unimodal", "boring"]

    def do_evaluate(self, x):
        x = x[0]
        return -(x - sin(x)) * exp(-(x**2))


class Problem21(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem21, self).__init__(dim)
        self.bounds = [(0, 10)]
        self.min_loc = 4.79507
        self.fmin = -9.50835
        self.fmax = 10.3367982489

    def do_evaluate(self, x):
        x = x[0]
        return x * sin(x) + x * cos(2 * x)


class Problem22(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem22, self).__init__(dim)
        self.bounds = [(0, 20)]
        self.min_loc = 9 * pi / 2
        self.fmin = exp(-27 * pi / 2) - 1
        self.fmax = 1.00000072495

    def do_evaluate(self, x):
        x = x[0]
        return exp(-3 * x) - (sin(x)) ** 3


problems = {
    # --- McCourt Family (High Dims -> Low Dims) ---
    "McCourt05": McCourt05(dim=12),
    "McCourt04": McCourt04(dim=10),
    # "McCourt03": McCourt03(dim=9),
    # "McCourt10": McCourt10(dim=8),
    # "McCourt11": McCourt11(dim=8),
    # "McCourt18": McCourt18(dim=8),
    # "McCourt25": McCourt25(dim=8),
    # "McCourt01": McCourt01(dim=7),
    # "McCourt02": McCourt02(dim=7),
    # "McCourt12": McCourt12(dim=7),
    # "McCourt17": McCourt17(dim=7),
    # "McCourt24": McCourt24(dim=7),
    # "McCourt07": McCourt07(dim=6),
    # "McCourt23": McCourt23(dim=6),
    # "McCourt06": McCourt06(dim=5),
    # "McCourt22": McCourt22(dim=5),
    # "McCourt08": McCourt08(dim=4),
    # "McCourt16": McCourt16(dim=4),
    # "McCourt21": McCourt21(dim=4),
    # "McCourt28": McCourt28(dim=4),
    # "McCourt09": McCourt09(dim=3),
    # "McCourt13": McCourt13(dim=3),
    # "McCourt14": McCourt14(dim=3),
    # "McCourt15": McCourt15(dim=3),
    # "McCourt26": McCourt26(dim=3),
    # "McCourt27": McCourt27(dim=3),
    # "McCourt19": McCourt19(dim=2),
    # "McCourt20": McCourt20(dim=2),
    # # --- High Dimensional Singletons ---
    "Cola": Cola(dim=17),
    "PowellTripleLog": PowellTripleLog(dim=12),
    "RosenbrockLog": RosenbrockLog(dim=11),
    "Pavianini": Pavianini(dim=10),
    "Xor": Xor(dim=9),
    "Levy03": Levy03(dim=8),
    "Rastrigin": Rastrigin(dim=8),
    "Trid": Trid(dim=6),
    "Watson": Watson(dim=6),
    "LennardJones6": LennardJones6(dim=6),
    "Hartmann6": Hartmann6(dim=6),
    "Dolan": Dolan(dim=5),
    # # --- Mixture of Gaussians Family ---
    "MixtureOfGaussians05": MixtureOfGaussians05(dim=8),
    "MixtureOfGaussians06": MixtureOfGaussians06(dim=8),
    # "MixtureOfGaussians01": MixtureOfGaussians01(dim=2),
    # "MixtureOfGaussians02": MixtureOfGaussians02(dim=2),
    # "MixtureOfGaussians03": MixtureOfGaussians03(dim=2),
    # "MixtureOfGaussians04": MixtureOfGaussians04(dim=2),
    # # --- Shekel Family ---
    # "Shekel05": Shekel05(dim=4),
    # "Shekel07": Shekel07(dim=4),
    # "Shekel10": Shekel10(dim=4),
    # # --- MegaDomain Family ---
    # "MegaDomain05": MegaDomain05(dim=4),
    # "MegaDomain02": MegaDomain02(dim=3),
    # "MegaDomain03": MegaDomain03(dim=3),
    # "MegaDomain04": MegaDomain04(dim=3),
    # "MegaDomain01": MegaDomain01(dim=2),
    # # --- Other 4D Problems ---
    # "Corana": Corana(dim=4),
    # "Gear": Gear(dim=4),
    # "Hartmann4": Hartmann4(dim=4),
    # "MieleCantrell": MieleCantrell(dim=4),
    # "Powell": Powell(dim=4),
    # "PowerSum": PowerSum(dim=4),
    # # --- 3D Problems ---
    # "BoxBetts": BoxBetts(dim=3),
    # "Hartmann3": Hartmann3(dim=3),
    # "HelicalValley": HelicalValley(dim=3),
    # "SchmidtVetters": SchmidtVetters(dim=3),
    # "Wolfe": Wolfe(dim=3),
    # # --- Schwefel Family (2D) ---
    # "Schwefel01": Schwefel01(dim=2),
    # "Schwefel06": Schwefel06(dim=2),
    # "Schwefel20": Schwefel20(dim=2),
    # "Schwefel22": Schwefel22(dim=2),
    # "Schwefel26": Schwefel26(dim=2),
    # "Schwefel36": Schwefel36(dim=2),
    # # --- Ursem Family (2D) ---
    # "Ursem01": Ursem01(dim=2),
    # "Ursem03": Ursem03(dim=2),
    # "Ursem04": Ursem04(dim=2),
    # "UrsemWaves": UrsemWaves(dim=2),
    # # --- Classic 2D Benchmarks (Alphabetical) ---
    # "Ackley": Ackley(dim=2),
    # "Adjiman": Adjiman(dim=2),
    # "Alpine01": Alpine01(dim=2),
    # "Alpine02": Alpine02(dim=2),
    # "ArithmeticGeometricMean": ArithmeticGeometricMean(dim=2),
    # "BartelsConn": BartelsConn(dim=2),
    # "Beale": Beale(dim=2),
    # "Bird": Bird(dim=2),
    # "Bohachevsky": Bohachevsky(dim=2),
    # "Branin01": Branin01(dim=2),
    # "Branin02": Branin02(dim=2),
    # "Brent": Brent(dim=2),
    # "Brown": Brown(dim=2),
    # "Bukin06": Bukin06(dim=2),
    # "CarromTable": CarromTable(dim=2),
    # "Chichinadze": Chichinadze(dim=2),
    # "Cigar": Cigar(dim=2),
    # "CosineMixture": CosineMixture(dim=2),
    # "CrossInTray": CrossInTray(dim=2),
    # "Csendes": Csendes(dim=2),
    # "Cube": Cube(dim=2),
    # "Damavandi": Damavandi(dim=2),
    # "Deb01": Deb01(dim=2),
    # "Deb02": Deb02(dim=2),
    # "Deceptive": Deceptive(dim=2),
    # "DeflectedCorrugatedSpring": DeflectedCorrugatedSpring(dim=2),
    # "DropWave": DropWave(dim=2),
    # "Easom": Easom(dim=2),
    # "EggCrate": EggCrate(dim=2),
    # "EggHolder": EggHolder(dim=2),
    # "ElAttarVidyasagarDutta": ElAttarVidyasagarDutta(dim=2),
    # "Exponential": Exponential(dim=2),
    # "Franke": Franke(dim=2),
    # "FreudensteinRoth": FreudensteinRoth(dim=2),
    # "Giunta": Giunta(dim=2),
    # "GoldsteinPrice": GoldsteinPrice(dim=2),
    # "Griewank": Griewank(dim=2),
    # "Hansen": Hansen(dim=2),
    # "HimmelBlau": HimmelBlau(dim=2),
    # "HolderTable": HolderTable(dim=2),
    # "Hosaki": Hosaki(dim=2),
    # "HosakiExpanded": HosakiExpanded(dim=2),
    # "JennrichSampson": JennrichSampson(dim=2),
    # "Judge": Judge(dim=2),
    # "Keane": Keane(dim=2),
    # "Langermann": Langermann(dim=2),
    # "Leon": Leon(dim=2),
    # "Levy05": Levy05(dim=2),
    # "Levy13": Levy13(dim=2),
    # "ManifoldMin": ManifoldMin(dim=2),
    # "Matyas": Matyas(dim=2),
    # "McCormick": McCormick(dim=2),
    # "Michalewicz": Michalewicz(dim=2),
    # "Mishra02": Mishra02(dim=2),
    # "Mishra06": Mishra06(dim=2),
    # "Mishra08": Mishra08(dim=2),
    # "Mishra10": Mishra10(dim=2),
    # "Ned01": Ned01(dim=2),
    # "Ned03": Ned03(dim=2),
    # "OddSquare": OddSquare(dim=2),
    # "Parsopoulos": Parsopoulos(dim=2),
    # "Penalty01": Penalty01(dim=2),
    # "Penalty02": Penalty02(dim=2),
    # "PenHolder": PenHolder(dim=2),
    # "Perm01": Perm01(dim=2),
    # "Perm02": Perm02(dim=2),
    # "Pinter": Pinter(dim=2),
    # "Plateau": Plateau(dim=2),
    # "Price": Price(dim=2),
    # "Qing": Qing(dim=2),
    # "Quadratic": Quadratic(dim=2),
    # "RippleBig": RippleBig(dim=2),
    # "RippleSmall": RippleSmall(dim=2),
    # "RosenbrockModified": RosenbrockModified(dim=2),
    # "Salomon": Salomon(dim=2),
    # "Sargan": Sargan(dim=2),
    # "Schaffer": Schaffer(dim=2),
    # "Shubert01": Shubert01(dim=2),
    # "Shubert03": Shubert03(dim=2),
    # "SineEnvelope": SineEnvelope(dim=2),
    # "SixHumpCamel": SixHumpCamel(dim=2),
    # "Sphere": Sphere(dim=2),
    # "Step": Step(dim=2),
    # "StretchedV": StretchedV(dim=2),
    # "StyblinskiTang": StyblinskiTang(dim=2),
    # "SumPowers": SumPowers(dim=2),
    # "TestTubeHolder": TestTubeHolder(dim=2),
    # "ThreeHumpCamel": ThreeHumpCamel(dim=2),
    # "Trefethen": Trefethen(dim=2),
    # "Tripod": Tripod(dim=2),
    # "VenterSobiezcczanskiSobieski": VenterSobiezcczanskiSobieski(dim=2),
    # "Weierstrass": Weierstrass(dim=2),
    # "XinSheYang02": XinSheYang02(dim=2),
    # "XinSheYang03": XinSheYang03(dim=2),
    # "YaoLiu": YaoLiu(dim=2),
    # "ZeroSum": ZeroSum(dim=2),
    # "Zimmerman": Zimmerman(dim=2),
    # # --- 1D Problems ---
    # "Problem02": Problem02(dim=1),
    # "Problem03": Problem03(dim=1),
    # "Problem04": Problem04(dim=1),
    # "Problem05": Problem05(dim=1),
    # "Problem06": Problem06(dim=1),
    # "Problem07": Problem07(dim=1),
    # "Problem09": Problem09(dim=1),
    # "Problem10": Problem10(dim=1),
    # "Problem11": Problem11(dim=1),
    # "Problem12": Problem12(dim=1),
    # "Problem13": Problem13(dim=1),
    # "Problem14": Problem14(dim=1),
    # "Problem15": Problem15(dim=1),
    # "Problem18": Problem18(dim=1),
    # "Problem20": Problem20(dim=1),
    # "Problem21": Problem21(dim=1),
    # "Problem22": Problem22(dim=1),
}

if __name__ == "__main__":
    for problem_name, problem in problems.items():
        print(problem_name)
        print(problem.bounds)
        print("-" * 100)
