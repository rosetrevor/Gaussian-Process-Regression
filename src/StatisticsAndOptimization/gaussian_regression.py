import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import sklearn.metrics
from typing import Callable, Any


def simulated_annealing(
    x_0: NDArray[Any],
    cost_fnc: Callable[[NDArray[Any]], float],
    temperature_0: float = 1000,
    mu: float = 0.99,
    step_size: float = 0.02,
) -> tuple[NDArray[Any], float]:
    """Simulated annealing algorithm.
    Reference: Kirkpatrick, Gelatt, Vecchi (1983), "Optimization by Simulated Annealing"

    Args:
        x_0 (NDArray): Starting point.
        cost_fnc (Callable): Function to evaluate.
        temperature_0 (float, optional): Initial temperature. Defaults to 1000.
        mu (float, optional): Rate of cooling after each iteration. Defaults to 0.99.
        step_size (float, optional): Step size to explore space. Defaults to .02.

    Returns:
        NDArray: Optimizer
        float: Optima
    """

    def metropolis_criteria(delta_f: float, _temperature: float) -> float:
        """The Metropolis criteria proposed in Kirkpatrick.

        Args:
            delta_f (float): Difference between function evaluations
            _temperature (float): Current temperature

        Returns:
            float: Metropolis criteria to evaluate against.
        """
        return np.exp(-delta_f / _temperature)

    def validate_x(x_vec: NDArray[Any]) -> NDArray[Any]:
        """Cholesky decomposition relies on positive semidefinite matrices.
        In some edge cases, a step can be taken which results in a matrix
        which is not positive semidefinite. Reset these to ensure parameters
        are greater than zero.

        Args:
            x_vec (NDArray): Current evaluation point.

        Returns:
            NDArray: Validated evaluation point.
        """
        # Ensure always positive
        if min(x_vec) <= 0:
            x_vec += min(x_vec) + 0.000001
        return x_vec

    def check_convergence(_temperature: float) -> bool:
        """Check if algorithm has converged.

        Args:
            _temperature (float): Current temperature

        Returns:
            bool: is converged
        """
        return temp <= 10

    # Kirkpatrick recommends T_k+1 = T_k * mu where 0 < mu < 1
    # Lundy-Mees propose T_k+1 = T_k / (1 + beta * T_k)
    # beta = (T_0 - T_f) / (m T_0 * T_f) where m steps are taken before updating

    # Initial testing seems Kirkpatrick is working okay, Lundy-Mees not implemented
    temp = temperature_0

    # Initialize at starting point
    x_star = x_0
    f_star = cost_fnc(x_0)

    x_old = x_0
    f_old = cost_fnc(x_0)

    converged = False
    while not converged:
        # Move randomly
        x_new = validate_x(x_old + (np.random.rand(3) * 2 - 1) * step_size)
        try:
            f_new = cost_fnc(x_new)
        except LinAlgError:
            f_new = f_old
            x_new = x_old
        if f_new <= f_star:
            # Always ccept step, update to new best
            x_star = x_new
            f_star = f_new

            x_old = x_new
            f_old = f_new
        else:
            del_f = f_old - f_new

            # Seek exploration and accept a worse step iff random number is below
            # Metropolis criteria. This happense more at high temps, as temps lower
            # the algorithm will favor exploitation as probability to take a worse
            # step becomes sufficiently low.
            if np.random.random() <= metropolis_criteria(del_f, temp):
                x_old = x_new
                f_old = f_new

        temp *= mu  # Cool the temperature after each iteration
        print(f"{temp:.2f}, {x_old=}, {f_old=:.2f}, {x_star=}, {f_star=:.2f}")
        if temp <= 10:
            # TODO: This convergence criteria could use refinement
            # A fixed temp is analagous to max number of iterations...
            converged = True
            print(f"CONVERGED: {x_star=}, {f_star=:.2f}")
            break
    return x_star, f_star


def normalization(
    x: NDArray[Any], y: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any], float, float]:
    """Normalize a set of x and y points. This is required for effective optimization as hyperparameters
    like step size will be significantly impacted.

    Args:
        x (NDArray): Array of x points
        y (NDArray): Array of y points

    Returns:
        NDArray: Normalized array of x points
        NDArray: Normalzied array of y points
        float: Scalar used to normalize x points
        float: Scalar used to normalized y points
    """
    scale_x = max(x) - min(x)
    scale_y = max(y) - min(y)
    return x / scale_x, y / scale_y, scale_x, scale_y


def kernel(
    a: NDArray[Any],
    b: NDArray[Any],
    length: float = 1.0,
    signal_variance2: float = 1.0,
    type: str = "SE",
) -> NDArray[Any]:
    """The kernel function used for Gaussian Process Regression. In this case,
    a square exponential covariance function is implemented.

    Args:
        a (float): Point a of the kernel
        b (float): Point b of the kernel
        length (float, optional): Length scale. Defaults to 1.
        signal_variance2 (float, optional): Signal variance squared. Defaults to 1.

    Returns:
        float: Kernel evaluation
    """
    if type == "SE":
        distance = sklearn.metrics.pairwise_distances(a, b)
        return signal_variance2 * np.exp(-1 / (2 * length**2) * distance**2)
    else:
        raise NotImplementedError(f"Kernel type of {type} is not supported")


def gaussian_process(
    x_train: NDArray[Any],
    y_train: NDArray[Any],
    x_test: NDArray[Any],
) -> tuple[
    Callable[[NDArray[Any]], tuple[NDArray[Any], NDArray[Any], float, NDArray[Any]]],
    Callable[[NDArray[Any]], float],
]:
    """Setup function for Gaussian Process Regression. This function returns a pair of callables. The callables are the following:
    - gaussian_process_regression (Callable): Returns mean regression, standard deviation, log marginal likelihood, and covariance matrix
    - cost_fnc (Callable): Returns only log marginal likelihood, intended use is for hyperparameter optimization

    Primary reference material: https://gaussianprocess.org/gpml/chapters/RW2.pdf

    Args:
        x_train (NDArray): Training data set
        y_train (NDArray): Training data set y points
        x_test (NDArray): Desired test points to evaluate

    Returns:
        Callable: Function to evaluate Gaussian Process Regression for the training data set on test data points
        as a function of a length scale, kernel noise parameter, and evaluation noise parameter.
        # TODO: Are kernel noise parameter and evaluation noise parameter consistent
    """

    def gaussian_process_regression(
        x_vec: NDArray[Any],
    ) -> tuple[NDArray[Any], NDArray[Any], float, NDArray[Any]]:
        """Gaussian Process Regression for a set of hyperparameters x_vec.

        Args:
            x_vec (NDArray): Vector containing length scale, signal variance, and noise variance

        Returns:
            NDArray: Gaussian process regression mean
            NDArray: Gaussian process regression standard deviation
            float: Log marginal likelihood
            NDArray: Covariance matrix
        """
        length: float = x_vec[0]
        signal_variance2: float = x_vec[1]
        noise_variance2: float = x_vec[2]  # This is technically the square

        # Reference Algorithm 2.1 of Rasmussen & Williams
        n: int = len(x_train)
        k_ss = kernel(x_test, x_test, length, signal_variance2)
        k = kernel(x_train, x_train, length, signal_variance2)
        l_cholesky = np.linalg.cholesky(k + noise_variance2 * np.eye(n))
        k_s = kernel(x_train, x_test, length, signal_variance2)

        alpha = np.linalg.solve(
            l_cholesky.transpose(), np.linalg.solve(l_cholesky, y_train)
        )
        mu = (k_s.transpose() @ alpha).squeeze()
        v = np.linalg.solve(l_cholesky, k_s)
        cov: NDArray[Any] = k_ss - v.transpose() @ v
        std_dev = np.sqrt(np.diag(cov)).squeeze()

        # Reference Equationq 2.30 of Rasmussen & Williams
        _lml_1 = -0.5 * y_train.transpose() @ alpha
        _lml_2 = -0.5 * np.log(np.linalg.det(l_cholesky))
        # _lml_3 = -n / 2 * np.log(2 * np.pi)  # This is const, could throw away
        log_marginal_likelihood = _lml_1[0][0] + _lml_2  # + _lml_3
        log_marginal_likelihood = -log_marginal_likelihood  # Negate for optimization
        return mu, std_dev, log_marginal_likelihood, cov

    def cost_fnc(x_vec: NDArray[Any]) -> float:
        """Cost function for hyperparameter optimization

        Args:
            x_vec (NDArray): Vector containing length scale, kernel noise, and evluation noise parameters

        Returns:
            float: Log marinal likelihood
        """
        # This allows re-write to optimize f(x)
        _, _, log_marginal_likelihood, cov = gaussian_process_regression(x_vec)
        return log_marginal_likelihood

    return gaussian_process_regression, cost_fnc


def main():
    def y(x: NDArray[Any], s: int, add_noise: bool = False):
        if add_noise:
            _noise = np.random.normal(0, noise, size=(s, 1))
        else:
            _noise = np.zeros((s, 1))
        return np.sin(x) + _noise
        # return np.sin(x) + x + _noise
        # return -((x + 2) ** 2) / +2 + _noise

    def plotter(
        _x_train: NDArray[Any],
        _y_train: NDArray[Any],
        _x_test: NDArray[Any],
        _y_test: NDArray[Any],
        _std_dev: NDArray[Any],
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
    y_test, std_dev, _, _ = gp_callable(x_star)
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
