import numpy as np


def sigmoid(xw: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-xw))

def predict(x: np.ndarray, w: np.ndarray, threshold: float=0.8, return_prob: bool=False):
    if len(x.shape) > 1:
        assert x.shape[1] == w.shape[0]
        xw = np.sum(x * w[np.newaxis, :], axis=1)
    else:
        assert x.shape[0] == w.shape[0]
        xw = np.sum(x * w)

    prob = sigmoid(xw=xw)
    if return_prob:
        return prob
    #print("prob: {}".format(prob))
    else:
        output = np.where(prob > threshold, 1, 0)
        return output

def sigmoid_cross_entropy_with_logits(xw: np.ndarray, y: np.ndarray) -> float:
    a = sigmoid(xw=xw)
    cost = - np.mean(y * np.log(a) + (1 - y) * np.log(1-a))
    return cost


def sigmoid_cross_entropy_with_x_w(x: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
    if len(x.shape) > 1:
        assert x.shape[1] == w.shape[0]
        xw = np.sum(x * w[np.newaxis, :], axis=1)
    else:
        assert x.shape[0] == w.shape[0]
        xw = np.sum(x * w)
    #return sigmoid_cross_entropy_with_logits(xw=xw, y=y)
    return sigmoid_cross_entropy_truncated(xw=xw, y=y)


def sigmoid_cross_entropy_truncated(xw: np.ndarray, y: np.ndarray):
    return np.mean(- y * xw + np.log(1+np.exp(xw)))


def derivative_cost_wrt_params(x: np.ndarray, w: np.ndarray, y: np.ndarray):
    #assert x.shape[0] == y.shape[0]
    if len(x.shape) > 1:
        assert x.shape[1] == w.shape[0]
        xw = np.sum(x * w[np.newaxis, :], axis=1)
        sig = sigmoid(xw=xw)
        return np.mean(- y[:, np.newaxis] * x + sig[:, np.newaxis] * x, axis=0)
    else:
        assert x.shape[0] == w.shape[0]
        xw = np.sum(x * w)
        sig = sig(xw=xw)
        return - y[np.newaxis] * x + sig[np.newaxis] * x


def backtracking_line_search(x: np.ndarray, w: np.ndarray, y: np.ndarray, p: np.ndarray, rho: float=0.9, alpha: float=5, c: float=1e-3) -> float:
    # Note that p = -grad we use + alpha, if p = grad we use - alpha
    gradient = derivative_cost_wrt_params(x=x, w=w, y=y)
    f_new = sigmoid_cross_entropy_with_x_w(x=x, w=w+alpha*p, y=y)
    f_old = sigmoid_cross_entropy_with_x_w(x=x, w=w, y=y)
    right_term = f_old + c * alpha * np.sum(gradient * p)
    #while f_new > f_old + c * alpha * np.sum(gradient * p):
    while f_new > right_term:
        alpha = rho * alpha
        #print("f_new: {}, f_old: {}, alpha: {}".format(f_new, f_old, alpha))
        f_new = sigmoid_cross_entropy_with_x_w(x=x, w=w+alpha*p, y=y)

    return alpha


def check_wolfe_II(x: np.ndarray, w: np.ndarray, y: np.ndarray, alpha: float, p: np.ndarray, c_2: float=0.9) -> bool:
    new_gradient = derivative_cost_wrt_params(x=x, w=w+alpha*p, y=y)
    gradient = derivative_cost_wrt_params(x=x, w=w, y=y)

    if np.sum(new_gradient * p) >= c_2 * np.sum(gradient * p):
        return True

    else:
        return False


def check_goldstein(x: np.ndarray, w: np.ndarray, y: np.ndarray, alpha: float, p: np.ndarray, c: float=0.25) -> bool:
    assert 0 < c < 0.5

    gradient = derivative_cost_wrt_params(x=x, w=w, y=y)
    f_new = sigmoid_cross_entropy_with_x_w(x=x, w=w+alpha*p, y=y)
    f_old = sigmoid_cross_entropy_with_x_w(x=x, w=w, y=y)

    if f_old + (1 - c) * alpha * np.sum(gradient * p) <= f_new and f_new <= f_old + c * alpha * np.sum(gradient * p):
        return True
    else:
        return False


#print(sigmoid_cross_entropy_with_x_w(x=np.array([[2, 1, 3], [-2, 1, -3]]), w=np.array([1, 1, 1]), y=np.array([1, 0])))
#print(sigmoid_cross_entropy_with_logits(xw=np.array([2, -1]), y=np.array([0, 1])))
#print(sigmoid_cross_entropy_truncated(xw=np.array([2, -1]), y=np.array([0, 1])))
#print(predict(x=np.array([[2, 1, 3], [-2, 1, -3]]), w=np.array([1, 1, 1]), threshold=0.8))
#print(derivative_cost_wrt_params(x=np.array([[2, 1, 3], [-2, 1, -3]]), w=np.array([1, 1, 1]), y=np.array([0, 1])))