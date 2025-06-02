import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    best_tau = 0
    min_MSE = 1
    for tau in tau_values:
        LWR = LocallyWeightedLinearRegression(tau)
        LWR.fit(x_train, y_train)
        y_pred = LWR.predict(x_eval)
        MSE = np.mean((y_pred - y_eval) ** 2)
        if MSE < min_MSE:
            min_MSE = MSE
            best_tau = tau

    # Fit a LWR model with the best tau value
    LWR_best = LocallyWeightedLinearRegression(best_tau)
    # Run on the test set to get the MSE value
    LWR_best.fit(x_train, y_train)
    y_test_pred = LWR_best.predict(x_test)
    y_pred = LWR_best.predict(x_eval)
    MSE_test = np.mean((y_test_pred - y_test) ** 2)
    np.savetxt(pred_path, y_test_pred)
    # Save predictions to pred_path
    # Plot data
    plt.figure()
    plt.title('tau = {}'.format(tau))
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_eval, y_pred, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05c_tau_{}.png'.format(tau))
    # *** END CODE HERE ***
