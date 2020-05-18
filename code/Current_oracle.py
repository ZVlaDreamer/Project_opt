import numpy as np

def F_Rozenbrock(x):
    #print(F)
    #F[0] = x[0] - 1
    #print(F)
    F = np.zeros_like(x)
    for i, x_i in enumerate(x):
        F[i] = x[i] - 1 if i == 0 else x_i + 1 - 2 * (x[i-1])**2
        #print(i)
        #F[i] = x_i + 1 - 2(x[i - 1])**2 + 1
    #print (F)
    return F

def grad_F_Rozenbrock(x):
    x_len = x.shape[0]
    grad_f = np.eye(x.shape[0])
    #print(grad_f)
    for i, x_i in enumerate(x):
        if(i+1 < x.shape[0]):
            grad_f[i+1][i] = -4 * x_i
    #print(grad_f)
    return grad_f

def psi_operator(y, x, f_x, L, grad_F):
    psi_1 = 1/2 * np.linalg.norm(f_x, 2) + L/2 * np.linalg.norm(y - x, 2)**2
    psi_2 = 1/(2 * np.linalg.norm(f_x, 2)) * np.linalg.norm(f_x + grad_F.dot(y - x), 2)**2
    return psi_1 + psi_2

def F_Rozenbrock_for_gd(x):
    return np.linalg.norm(F_Rozenbrock(x), 2)**2
def grad_Rozenbrock_for_gd(x):
    grad_f = np.zeros_like(x)
    
    for i, x_i in enumerate(x):
        grad_f[i] = 1/2 * (x_i - 1) - 8 * x_i * (x[i + 1] - 2 * x_i**2 + 1) if i == 0 else 2 * (x_i - 2 * x[i - 1]**2 + 1) 
        - 8 * x_i * (x[i + 1] - 2 * x_i**2 + 1) if i < x.shape[0] - 1 else 2 * (x_i - 2 * x[i - 1]** 2 + 1)
    
    return grad_f



def F_Trigonom(x):
    n = x.shape[0]
    F = np.zeros(n)
    F[0] = x[0] - 1
    for i in range(n-1):
        F[i+1] = x[i+1] + np.cos(np.pi * x[i])
    return F

def jacobian_F(x):
    n = x.shape[0]
    J = np.zeros((n, n))
    J[0,0] = 1
    for i in range(n-1):
        J[i+1,i+1] = 1
        J[i+1,i] = - np.pi * np.sin(np.pi * x[i-1])
    return J
