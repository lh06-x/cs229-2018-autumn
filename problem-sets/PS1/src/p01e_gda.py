import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    model = GDA()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        y__1 = sum(y == 1)
        y__0 = sum(y == 0)

        phi = (1 / m) * y__1
        miu_0 = sum((y == 0) * x) / y__0
        miu_1 = sum((y == 1) * x) / y__1
        sigma = ((x - miu_0).dot((x - miu_0).T) + (x - miu_1).dot((x - miu_1).T)) / m
        sigma_inv = np.linalg.inv(sigma)

        self.theta[1:] = sigma_inv.dot(miu_1 - miu_0)
        self.theta[0] = (((miu_0 + miu_1).T.dot(sigma_inv).dot(miu_0 - miu_1)) / 2) - np.log((1 - phi) / phi)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-(self.theta[1:].dot(x) + self.theta[0])))
        # *** END CODE HERE
