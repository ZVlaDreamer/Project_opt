import numpy as np






class Oracle_Trigonometric_Oscillator():
    def __init__(self, n):
        self.n = n

    def F(self, x):
        F = np.zeros(self.n)
        F[0] = x[0] - 1
        for i in range(self.n-1):
            F[i+1] = x[i+1] + np.cos(np.pi * x[i])
        return F

    def f1(self, x):
        return np.sqrt(np.sum(self.F(x) ** 2))

    def hat_f1(self, x):
        return self.f1(x) / np.sqrt(self.n)

    def jacobian_F(self, x):
        J = np.zeros((self.n, self.n))
        J[0,0] = 1
        for i in range(self.n-1):
            J[i+1,i+1] = 1
            J[i+1,i] = - np.pi * np.sin(np.pi * x[i-1])
        return J

    def grad_hat_f1(self, x):
        g = np.zeros(self.n)
        g[0] = 2 * (x[0] - 1) - 2 * np.pi * np.sin(np.pi * x[0]) * (x[1] + np.cos(np.pi * x[0]))
        for i in range(self.n-2):
            g[i+1] = 2 * (x[i+1] + np.cos(np.pi * x[i]))  - 2 * np.pi * np.sin(np.pi * x[i+1]) * (x[i+2] + np.cos(np.pi * x[i+1]))

        g[self.n - 1] = 2 * (x[self.n-1] + np.cos(np.pi * x[self.n-2]))
        return g / (2 * self.n * self.hat_f1(x))



class Oracle_Chebyshev_Oscillator():
    def __init__(self, n):
        self.n = n

    def F(self, x):
        F = np.zeros(self.n)
        F[0] = x[0] - 1
        for i in range(self.n-1):
            F[i+1] = x[i+1] - 2 * x[i] ** 2 + 1
        return F

    def f1(self, x):
        return np.sqrt(np.sum(self.F(x) ** 2))

    def hat_f1(self, x):
        return self.f1(x) / np.sqrt(self.n)

    def jacobian_F(self, x):
        J = np.zeros((self.n, self.n))
        J[0,0] = 1
        for i in range(self.n-1):
            J[i+1,i+1] = 1
            J[i+1,i] = - 4 * x[i]
        return J

    def grad_hat_f1(self, x):
        g = np.zeros(self.n)
        g[0] = 2 * (x[0] - 1) - 8 * x[0] * (x[1] - 2 * x[0] ** 2 + 1)
        for i in range(self.n-2):
            g[i+1] = 2 * (x[i+1] - 2 * x[i] ** 2 + 1) - 8 * x[i+1] * (x[i+2] - 2 * x[i+1] ** 2 + 1)

        g[self.n-1] = 2 * (x[self.n-1] - 2 * x[self.n-2] ** 2 + 1)
        return g / (2 * self.n * self.hat_f1(x))
