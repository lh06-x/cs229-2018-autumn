import numpy as np
import util

from linear_model import LinearModel
from notes.backprop import sigmoid


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, x, y):
        h = self.sigmoid(x.dot(self.theta))
        m = x.shape[0]
        cost = - (1 / m) * (y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h)))
        return cost

    def compute_gradient(self, x, y):
        h = self.sigmoid(x.dot(self.theta))
        m = x.shape[0]
        grad = (1 / m) * (x.T.dot(h - y))
        return grad

    def compute_hessian(self, x, y):
        h = self.sigmoid(x.dot(self.theta))
        m = x.shape[0]
        hessian = (1 / m) * (x.T * h * (1 - h).dot(x))
        return hessian

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        n_features = x.shape[1]
        self.theta = np.random.normal(0, 0.01, n_features)

        for i in range(self.max_iter):
            cost = self.compute_cost(x, y)
            grad = self.compute_gradient(x, y)
            hessian = self.compute_hessian(x, y)
            new_theta = np.linalg.solve(hessian, grad)
            self.theta = self.theta - new_theta
            if np.linalg.norm(new_theta - self.theta) < self.eps:
                print(f"Converged after {i + 1} iterations")
                break

        self.theta = new_theta
        return self

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        prob = self.sigmoid(x.dot(self.theta))
        return (prob >= 0.5).astype(int)
        # *** END CODE HERE ***
