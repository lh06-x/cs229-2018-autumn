import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    x_train, t_train = util.load_dataset(train_path, label_col= "t", add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col="t", add_intercept=True)

    model = LogisticRegression()
    model.fit(x_train, t_train)

    pred_t_c = model.predict(x_test)
    util.plot(x_test, t_test, model.theta, 'output/p02c.png')
    np.savetxt(pred_path_c, pred_t_c > 0.5, fmt='%d')
    # Make sure to save outputs to pred_path_c
    # Part (d): Train on y-labels and test on true labels
    x_train, y_train = util.load_dataset(train_path, label_col="y", add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col= "y", add_intercept=True)

    model_y = LogisticRegression()
    model_y.fit(x_train, y_train)

    pred_y_d = model_y.predict(x_test)
    np.savetxt(pred_path_d, pred_y_d > 0.5, fmt ='%d')
    util.plot(x_test, y_test, model_y.theta, 'output/p02d.png')
    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    x_eval, y_eval = util.load_dataset(valid_path, label_col="y", add_intercept=True)
    alpha = np.mean(model_y.predict(x_eval))
    correction = 1 + np.log(2 / alpha - 1) / model_y.theta[0]

    util.plot(x_test, y_test, model_y.theta, 'output/p02e.png', correction = correction)
    t_pred_e = y_test / alpha
    np.savetxt(pred_path_e, t_pred_e > 0.5, 'output/p02e.png')
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
