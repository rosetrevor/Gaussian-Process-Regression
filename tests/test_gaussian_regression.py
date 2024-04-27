import numpy as np

from StatisticsAndOptimization.gaussian_regression import (
    simulated_annealing,
    _check_convergence,
    _validate_x,
    metropolis_criteria,
)


def test_simulated_annealing():
    simulated_annealing
    assert True is True


def test_metropolis_criteria():
    assert True is True


def test_convergence():
    assert _check_convergence(15) is False
    assert _check_convergence(8) is True


def test_metropolis():
    initial_temp = 10
    assert metropolis_criteria(0.1, initial_temp) < initial_temp


def test_x_validation():
    x = np.array([2.0, 1.0, 2.1])
    assert (_validate_x(x) == x).all()

    validated_x = True
    x = np.array([2.0, -1.0, 0.0])
    for ret_val in _validate_x(x):
        if ret_val < 0.0:
            validated_x = False
    assert validated_x


def main():
    test_simulated_annealing()
    test_x_validation()


if __name__ == "__main__":
    main()
