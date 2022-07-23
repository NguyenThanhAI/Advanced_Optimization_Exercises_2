from typing import List, Tuple
import numpy as np


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def exponential_moving_average(signal: List, weight: float) -> np.ndarray:
    ema = np.zeros(len(signal))
    ema[0] = signal[0]

    for i in range(1, len(signal)):
        ema[i] = (signal[i] * weight) + (ema[i - 1] * (1 - weight))

    return ema
    

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
    cost = - np.mean(y * np.log(a + 1e-8) + (1 - y) * np.log(1-a + 1e-8))
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
    #right_term = f_old + c * alpha * np.sum(gradient * p)
    while f_new > f_old + c * alpha * np.sum(gradient * p):
    #while f_new > right_term:
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


def quadratic_interplolate(x: np.ndarray, w: np.ndarray, y: np.ndarray, p: np.ndarray, alpha_low: float, alpha_high: float) -> float:
    assert alpha_low < alpha_high
    gradient_phi_alpha_low = derivative_cost_wrt_params(x=x, w=w+alpha_low*p, y=y)
    derivative_phi_alpha_low = np.sum(gradient_phi_alpha_low * p)
    phi_alpha_low = sigmoid_cross_entropy_with_x_w(x=x, w=w+alpha_low*p, y=y)
    phi_alpha_high = sigmoid_cross_entropy_with_x_w(x=x, w=w+alpha_high*p, y=y)
    alpha_j = alpha_low - ((alpha_low - alpha_high) * derivative_phi_alpha_low)/(2 * (derivative_phi_alpha_low - ((phi_alpha_low - phi_alpha_high)/(alpha_low - alpha_high))))
    return alpha_j


def zoom(x: np.ndarray, w: np.ndarray, y: np.ndarray, p: np.ndarray, alpha_low: float, alpha_high: float, c_1: float=0.0001, c_2: float=0.9) -> float:
    gradient_zero = derivative_cost_wrt_params(x=x, w=w, y=y)
    gradient_phi_zero = np.sum(gradient_zero * p)
    phi_zero = sigmoid_cross_entropy_with_x_w(x=x, w=w, y=y)

    while True:
        alpha_j = quadratic_interplolate(x=x, w=w, y=y, p=p, alpha_low=alpha_low, alpha_high=alpha_high)
        phi_alpha_j = sigmoid_cross_entropy_with_x_w(x=x, w=w+alpha_j*p, y=y)
        phi_alpha_low = sigmoid_cross_entropy_with_x_w(x=x, w=w+alpha_low*p, y=y)
        #phi_alpha_high = sigmoid_cross_entropy_with_x_w(x=x, w=w+alpha_high*p, y=y)

        if phi_alpha_j > (phi_zero + c_1 * alpha_j * gradient_phi_zero) or (phi_alpha_j >= phi_alpha_low):
            #print("If")
            alpha_high = alpha_j
        
        else:
            #print("Else")
            gradient_alpha_j = derivative_cost_wrt_params(x=x, w=w+alpha_j*p, y=y)
            gradient_phi_alpha_j = np.sum(gradient_alpha_j * p)
            if np.abs(gradient_phi_alpha_j) <= - c_2 * gradient_phi_zero:
                return alpha_j

            if gradient_phi_alpha_j * (alpha_high - alpha_low) >= 0:
                alpha_high = alpha_low

            alpha_low = alpha_j


def line_search(x: np.ndarray, w: np.ndarray, y: np.ndarray, p: np.ndarray, alpha_max: float=1, c_1: float=0.0001, c_2: float=0.9) -> float:
    prev_alpha = 0
    present_alpha = 0.9 * alpha_max

    gradient_zero = derivative_cost_wrt_params(x=x, w=w, y=y)
    gradient_phi_zero = np.sum(gradient_zero * p)

    #prev_gradient_alpha = derivative_cost_wrt_params(x=x, w=w+prev_alpha*p, y=y)
    #prev_gradient_phi_alpha = np.sum(prev_gradient_alpha * p)

    phi_zero = sigmoid_cross_entropy_with_x_w(x=x, w=w, y=y)
    prev_phi_alpha = sigmoid_cross_entropy_with_x_w(x=x, w=w+prev_alpha*p, y=y)

    i = 1

    while True:

        present_phi_alpha = sigmoid_cross_entropy_with_x_w(x=x, w=w+present_alpha*p, y=y)
        if (present_phi_alpha > phi_zero + c_1 * present_alpha * gradient_phi_zero) or ((present_phi_alpha >= prev_phi_alpha) and (i > 1)):
            alpha_star = zoom(x=x, w=w, y=y, p=p, alpha_low=prev_alpha, alpha_high=present_alpha, c_1=c_1, c_2=c_2)
            return alpha_star

        present_gradient_alpha = derivative_cost_wrt_params(x=x, w=w+present_alpha*p, y=y)
        present_gradient_phi_alpha = np.sum(present_gradient_alpha * p)

        if np.abs(present_gradient_phi_alpha) <= - c_2 * gradient_phi_zero:
            alpha_star = present_alpha
            return alpha_star
            
        if present_gradient_phi_alpha >= 0:
            alpha_star = zoom(x=x, w=w, y=y, p=p, alpha_low=present_alpha, alpha_high=prev_alpha, c_1=c_1, c_2=c_2)
            return alpha_star

        new_alpha = (present_alpha + alpha_max) / 2
        prev_alpha = present_alpha
        present_alpha = new_alpha

        prev_phi_alpha = present_phi_alpha

        i += 1


def adjust_step_length(x: np.ndarray, w: np.ndarray, y: np.ndarray, alpha: float, p: np.ndarray, epsilon: float=1e-3, delta=1e-6):
    phi_alpha = sigmoid_cross_entropy_with_x_w(x=x, w=w+alpha*p, y=y)
    phi_alpha_plus_epsilon = sigmoid_cross_entropy_with_x_w(x=x, w=w+(alpha + epsilon) * p, y=y)
    phi_alpha_subtract_epsilon = sigmoid_cross_entropy_with_x_w(x=x, w=w+(alpha - epsilon) * p, y=y)
    phi_alpha_plus_2epsilon = sigmoid_cross_entropy_with_x_w(x=x, w=w+(alpha + 2 * epsilon) * p, y=y)
    phi_alpha_subtract_2epsilon = sigmoid_cross_entropy_with_x_w(x=x, w=w+(alpha - 2 * epsilon) * p, y=y)

    new_alpha = alpha - 2 * epsilon * (phi_alpha_plus_epsilon - phi_alpha_subtract_epsilon) / (phi_alpha_plus_2epsilon + phi_alpha_subtract_2epsilon - 2 * phi_alpha + delta)

    return new_alpha


def adam_step(weights: np.ndarray, dweights: np.ndarray, m: np.ndarray, v: np.ndarray, alpha: float, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta_1 * m + (1 - beta_1) * dweights
    v = beta_2 * v + (1 - beta_2) * dweights**2

    m_hat = m / (1 - beta_1**t)
    v_hat = v / (1 - beta_2**t)

    weights = weights - (alpha * m_hat) / (np.sqrt(v_hat) + epsilon)

    return weights, m, v


def adamax_step(weights: np.ndarray, dweights: np.ndarray, m: np.ndarray, u: float, alpha: float, t: int, beta_1: float=0.9, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, float]:

    m = beta_1 * m + (1 - beta_1) * dweights
    u = np.maximum(beta_2 * u, np.max(dweights))

    weights = weights - (alpha * m) / ((1 - beta_1**t) * u + epsilon)

    return weights, m, u


def nadam_step(weights: np.ndarray, dweights: np.ndarray, m: np.ndarray, v: np.ndarray, alpha: float, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta_1 * m + (1 - beta_1) * dweights
    v = beta_2 * v + (1 - beta_2) * dweights**2

    m_hat = m / (1 - beta_1**t)
    v_hat = v / (1 - beta_2**t)
    
    weights = weights - (alpha * (beta_1 * m_hat + ((1 - beta_1)/(1 - beta_1**t))*dweights)) / (np.sqrt(v_hat) + epsilon)

    return weights, m, v


def amsgrad_step(weights: np.ndarray, dweights: np.ndarray, m: np.ndarray, v: np.ndarray, v_hat: np.ndarray, alpha: float, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = beta_1 * m + (1 - beta_1) * dweights
    v = beta_2 * v + (1 - beta_2) * dweights**2

    v_hat = np.maximum(v_hat, v)

    weights = weights - (alpha * m) / (np.sqrt(v_hat) + epsilon)

    return weights, m, v, v_hat


def adabelief_step(weights: np.ndarray, dweights: np.ndarray, m: np.ndarray, v: np.ndarray, alpha: float, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta_1 * m + (1 - beta_1) * dweights
    v = beta_2 * v + (1 - beta_2) * (dweights - m) ** 2

    m_hat = m / (1 - beta_1**t)
    v_hat = v / (1 - beta_2**t)

    weights = weights - (alpha * m_hat) / (np.sqrt(v_hat) + epsilon)

    return weights, m, v


def adagrad_step(weights: np.ndarray, dweights: np.ndarray, v: np.ndarray, alpha: float, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray]:
    v = v + dweights**2

    weights = weights - (alpha * dweights) / (np.sqrt(v) + epsilon)

    return weights, v


def rmsprop_step(weights: np.ndarray, dweights: np.ndarray, v: np.ndarray, alpha: float, beta_2: float=0.9, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray]:
    v = beta_2 * v + (1 - beta_2) * dweights ** 2

    weights = weights - (alpha * dweights) / (np.sqrt(v) + epsilon)

    return weights, v


def momentum_step(weights: np.ndarray, dweights: np.ndarray, m: np.ndarray, alpha: float, beta_1: float=0.9) -> Tuple[np.ndarray, np.ndarray]:
    m = beta_1 * m + (1 - beta_1) * dweights

    weights = weights - alpha * m

    return weights, m


def adadelta_step(weights: np.ndarray, dweights: np.ndarray, v: np.ndarray, d: np.ndarray, alpha: float, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = beta_2 * v + (1 - beta_2) * dweights ** 2

    delta_w = - (alpha * np.sqrt(d + epsilon) * dweights) / (np.sqrt(v + epsilon))

    weights = weights + delta_w

    d = beta_2 * d + (1 - beta_2) * delta_w ** 2

    return weights, v, d


#print(sigmoid_cross_entropy_with_x_w(x=np.array([[2, 1, 3], [-2, 1, -3]]), w=np.array([1, 1, 1]), y=np.array([1, 0])))
#print(sigmoid_cross_entropy_with_logits(xw=np.array([2, -1]), y=np.array([0, 1])))
#print(sigmoid_cross_entropy_truncated(xw=np.array([2, -1]), y=np.array([0, 1])))
#print(predict(x=np.array([[2, 1, 3], [-2, 1, -3]]), w=np.array([1, 1, 1]), threshold=0.8))
#print(derivative_cost_wrt_params(x=np.array([[2, 1, 3], [-2, 1, -3]]), w=np.array([1, 1, 1]), y=np.array([0, 1])))

'''x=np.array([[2, 1, 3], [-2, 1, -3]])
w=np.array([1, 1, 1])
y=np.array([0, 1])

alpha_low = 1
alpha_high = 2

c_1 = 0.0001
c_2 = 0.9
c = 0.25

dweight = derivative_cost_wrt_params(x=x, w=w, y=y)

alpha_j = quadratic_interplolate(x=x, w=w, y=y, p=-dweight, alpha_low=alpha_low, alpha_high=alpha_high)
alpha_star = zoom(x=x, w=w, y=y, p=-dweight, alpha_low=alpha_low, alpha_high=alpha_high, c_1=c_1, c_2=c_2)
alpha = line_search(x=x, w=w, y=y, p=-dweight, alpha_max=alpha_high, c_1=c_1, c_2=c_2)
print(alpha_j, alpha_star, alpha)
print(check_wolfe_II(x=x, w=w, y=y, alpha=alpha, p=-dweight, c_2=c_2), check_goldstein(x=x, w=w, y=y, alpha=alpha, p=-dweight, c=c))'''