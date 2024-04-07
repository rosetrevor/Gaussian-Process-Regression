import numpy as np
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
import sklearn.metrics
import scipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time


def simulated_annealing(
    x_0: np.array,
    cost_fnc: callable,
    temperature_0: float = 1000,
    mu: int = 0.99,
    b: float = 50,
):
    def metropolis_criteria(delta_f, _temperature):
        return np.exp(-delta_f / _temperature)

    def validate_x(x_vec):
        # Ensure always positive
        if min(x_vec) <= 0:
            x_vec += min(x_vec) + 0.000001
        return x_vec

    # Kirkpatric recommends T_k+1 = T_k * mu where 0 < mu < 1
    # Lundy-Mees propose T_k+1 = T_k / (1 + beta * T_k)
    # beta = (T_0 - T_f) / (m T_0 * T_f) where m steps are taken before updating

    temp = temperature_0

    # Initialize at starting point
    x_star = x_0
    f_star = cost_fnc(x_0)

    x_old = x_0
    f_old = cost_fnc(x_0)

    converged = False
    while not converged:
        # Move randomly
        x_new = validate_x(x_old + (np.random.rand(3) * 2 - 1) / b)
        try:
            f_new = cost_fnc(x_new)
        except LinAlgError:
            f_new = f_old
            x_new = x_old
        if f_new <= f_star:
            # Accept step, update to new best
            x_star = x_new
            f_star = f_new

            x_old = x_new
            f_old = f_new
        else:
            z = np.random.random()
            del_f = f_old - f_new
            if np.random.random() <= metropolis_criteria(del_f, temp):
                # Accept a step in a worse direction for exploration
                x_old = x_new
                f_old = f_new

        temp *= mu
        print(f"{temp:.2f}, {x_old=}, {f_old=:.2f}, {x_star=}, {f_star=:.2f}")
        if temp <= 10:
            converged = True
            print(f"CONVERGED: {x_star=}, {f_star=:.2f}")
            return x_star, f_star


def normalization(x, y):
    scale_x = max(x) - min(x)
    scale_y = max(y) - min(y)
    return (x - min(x)) / scale_x, y / scale_y, scale_x, scale_y


def kernel(a, b, length=1, sigma_f=1):
    distance = sklearn.metrics.pairwise_distances(a, b)
    return sigma_f * np.exp(-1 / (2 * length**2) * distance**2)


def gaussian_process(
    x_train: np.array,
    y_train: np.array,
    x_test: np.array,
) -> callable:
    # Curry the optimization so it can be called as a vector
    def gaussian_process_regression(x_vec):
        # https://gaussianprocess.org/gpml/chapters/RW2.pdf
        length: float = x_vec[0]
        sigma_f: float = x_vec[1]
        sigma_n2: float = x_vec[2]  # This is technically the square

        n = len(x_train)
        k_ss = kernel(x_test, x_test, length, sigma_f)
        k = kernel(x_train, x_train, length, sigma_f)
        l = np.linalg.cholesky(k + sigma_n2 * np.eye(n))
        k_s = kernel(x_train, x_test, length, sigma_f)

        alpha = np.linalg.solve(l.transpose(), np.linalg.solve(l, y_train))
        mu = (k_s.transpose() @ alpha).squeeze()
        v = np.linalg.solve(l, k_s)
        cov = k_ss - v.transpose() @ v
        std_dev = np.sqrt(np.diag(cov)).squeeze()

        _lml_1 = -0.5 * y_train.transpose() @ alpha
        _lml_2 = -0.5 * np.log(np.linalg.det(l))
        # _lml_3 = -n / 2 * np.log(2 * np.pi)  # This is const, could throw away
        log_marginal_likelihood = _lml_1[0][0] + _lml_2  # + _lml_3
        log_marginal_likelihood = -log_marginal_likelihood  # Negate for optimization
        return mu, std_dev, log_marginal_likelihood

    def cost_fnc(x_vec):
        # This allows re-write to optimize f(x)
        _, _, log_marginal_likelihood = gaussian_process_regression(x_vec)
        return log_marginal_likelihood

    return gaussian_process_regression, cost_fnc


def main():
    def y(x: np.array, s: int, add_noise: bool = False):
        if add_noise:
            _noise = np.random.normal(0, noise, size=(s, 1))
        else:
            _noise = np.zeros((s, 1))
        # return np.sin(x) + _noise
        # return np.sin(x) + x + _noise
        return -((x + 2) ** 2) + 2 + _noise

    def plotter(
        _x_train: np.array,
        _y_train: np.array,
        _x_test: np.array,
        _y_test: np.array,
        _std_dev: np.array,
    ) -> None:
        # plt.plot(_x_test, y(_x_test, len(_x_test)))
        plt.plot(_x_train, _y_train, "bx", ms=6)
        # plt.plot(_x_test.flat, _y_test - 1 * _std_dev, ":", color="#080808")
        # plt.plot(_x_test.flat, _y_test + 1 * _std_dev, ":", color="#080808")
        # plt.plot(_x_test.flat, _y_test - 2 * _std_dev, "--", color="#080808")
        # plt.plot(_x_test.flat, _y_test + 2 * _std_dev, "--", color="#080808")
        # plt.plot(_x_test.flat, _y_test - 3 * _std_dev, color="#080808")
        # plt.plot(_x_test.flat, _y_test + 3 * _std_dev, color="#080808")
        plt.gca().fill_between(
            _x_test.flat,
            _y_test - 3 * _std_dev,
            _y_test + 3 * _std_dev,
            color="#dddddd",
        )
        plt.plot(_x_test, _y_test, "r--", lw=2)
        plt.title("Three samples from GP posterior")
        plt.show()

    n_train = 100
    x_train = np.random.normal(0, 3, size=(n_train, 1))
    noise = 0.2
    y_train = y(x_train, n_train, add_noise=True)

    x_train_norm, y_train_norm, x_scale, y_scale = normalization(x_train, y_train)

    # x_train_norm = x_train
    # y_train_norm = y_train
    # x_scale = 1.0
    # y_scale = 1.0

    n_test = 50
    x_test = np.linspace(min(x_train_norm), max(x_train_norm), n_test).reshape(-1, 1)

    gp_callable, cost_fnc = gaussian_process(x_train_norm, y_train_norm, x_test)
    x_star, f_star = simulated_annealing(np.array([0.1, 0.1, 0.05]), cost_fnc)
    y_test, std_dev, _ = gp_callable(x_star)
    plotter(
        x_train_norm,
        y_train_norm,
        x_test,
        y_test,
        std_dev,
    )
    # for lscale in [1, 5, 0.1]:
    #     y_test, std_dev, lml = gaussian_process(np.array([lscale, 1, 1]))
    #     print(lml)
    #     plotter(x_train, y_train, x_test, y_test, std_dev)


if __name__ == "__main__":
    main()
