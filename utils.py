import numpy as np
from cvxopt import matrix, solvers


def project(x, lower, upper):
    return np.clip(x, lower, upper)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def armijo_line_search(f, current_grad, d, x, delta_init=1.0, gamma=0.1, delta_factor=0.9, max_iter=10000):

    alpha = delta_init
    f_x = f(x)
    for _ in range(max_iter):
        f_new = f(x + alpha * d)
        if f_new <= f_x + alpha * gamma * np.dot(current_grad, d):
            return alpha
        alpha *= delta_factor
    return alpha

def projected_gradient_descent(f, grad, x0, lower, upper, epsilon=1e-3, max_iter=10000):

    x = x0
    for i in range(max_iter):
        current_grad = grad(x)
        x_new = project(x - current_grad, lower, upper)
        d = x_new - x
        if np.linalg.norm(d) < epsilon:
            return {"f": f(x), "iterations": i + 1}
        alpha = armijo_line_search(f, current_grad, d, x)
        x = x + alpha * d
    return {"f": f(x), "iterations": max_iter}

def find_newton_direction(current_grad, current_hess, x, lower, upper):
    n = len(x)
    Q = matrix(current_hess)
    p = matrix((current_grad - current_hess @ x).astype(np.double))

    G_list = []
    h_list = []
    
    for i in range(n):
        if upper[i] < 1e15: 
            row = np.zeros(n)
            row[i] = 1.0
            G_list.append(row)
            h_list.append(upper[i])
        if lower[i] > -1e15:
            row = np.zeros(n)
            row[i] = -1.0
            G_list.append(row)
            h_list.append(-lower[i])
            
    if not G_list:
        G = None
        h = None
    else:
        G = matrix(np.vstack(G_list).astype(np.double))
        h = matrix(np.array(h_list).astype(np.double))
    
    solvers.options['show_progress'] = False
    
    try:
        sol = solvers.qp(Q, p, G, h)
        if sol['status'] != 'optimal':
            return None 
        found_x = np.array(sol['x']).flatten()
        return found_x - x
    except Exception:
        return None 

def projected_newton(f, grad, hess, x0, lower, upper, epsilon=1e-3, max_iter=10000):

    x = x0
    newton_iters = 0
    for i in range(max_iter):
        current_grad = grad(x)
        current_hess = hess(x)
        if is_pos_def(current_hess):
            newton_iters += 1
            d = find_newton_direction(current_grad, current_hess, x, lower, upper)
        else:
            x_new = project(x - current_grad, lower, upper)
            d = x_new - x
        if np.linalg.norm(x - project(x - current_grad, lower, upper)) < epsilon:
            return {"f": f(x), "iterations": i + 1, "newton_iterations": newton_iters}
        alpha = armijo_line_search(f, current_grad, d, x)
        x = x + alpha * d
    return {"f": f(x), "iterations": max_iter, "newton_iterations": newton_iters}

def modified_projected_newton(f, grad, hess, x0, m, lower, upper, epsilon=1e-3, max_iter=10000):
    x = x0
    H = np.eye(len(x0))
    newton_iters = 0
    found_newton_region = False
    for i in range(max_iter):
        current_grad = grad(x)
        if(i % m == 0):
            current_hess = hess(x)
            if is_pos_def(current_hess):
                H = current_hess
                found_newton_region = True
        if found_newton_region:
            newton_iters += 1
            d = find_newton_direction(current_grad, H, x, lower, upper)
            if np.linalg.norm(d) < 1e-8:
                found_newton_region = False
        else:
             x_new = project(x - current_grad, lower, upper)
             d = x_new - x
        current_hess = hess(x)
        if np.linalg.norm(np.linalg.norm(x - project(x - current_grad, lower, upper))) < epsilon:
            return {"f": f(x), "iterations": i + 1, "newton_iterations": newton_iters}
        alpha = armijo_line_search(f, current_grad, d, x)
        x = x + alpha * d
    return {"f": f(x), "iterations": max_iter, "newton_iterations": newton_iters}