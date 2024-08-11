import os
import numpy as np
from multiprocessing import Pool


class NCS:

    def __init__(self, f: callable, num_pop: int, lb: np.ndarray, ub: np.ndarray, max_iter: int, r: np.float64, epoch: int, id: int) -> None:
        self.f = f
        self.num_pop = num_pop
        self.lb = lb
        self.ub = ub
        self.max_iter = max_iter
        self.r = r
        self.epoch = epoch
        self.id = id
        self.n = len(self.lb)

        self.sigma_init = np.array((self.ub - self.lb) / self.num_pop)
        self.sigma = np.ones((self.num_pop, self.n)) * self.sigma_init / 10

        self.params_father = np.random.uniform(low=self.lb, high=self.ub, size=(self.num_pop, self.n))
        self.reward_father = np.empty((self.num_pop, 1))
        self.params_child = np.empty((self.num_pop, self.n))
        self.reward_child = np.empty((self.num_pop, 1))

        self.best_score = 0
        self.best_param = np.empty(self.n)

        self.update_count = np.zeros((self.num_pop, 1))
        self.iter = 0

        if not os.path.exists('log'):
            os.mkdir('log')

    def generate_child(self) -> None:
        self.params_child = self.params_father + np.random.normal(scale=self.sigma)
        self.params_child = np.clip(self.params_child, self.lb, self.ub)

    def update_best(self) -> None:
        best = np.min(self.reward_child)
        if best < self.best_score:
            self.best_score = best
            self.best_param = self.params_child[np.argmin(self.reward_child)]

    def update_lambda(self) -> None:
        self.lambda_t = np.random.normal(1, 0.1 - 0.1 * self.iter / self.max_iter)

    def minmax_param(self, param) -> np.array:
        return (param - self.lb) / (self.ub - self.lb)

    def minmax_sigma(self, sigma) -> np.array:
        return sigma / self.sigma_init

    def Bhattacharyya_Distance(self, params, id_1, id_2) -> np.ndarray:
        param_1 = self.minmax_param(params[id_1])
        param_2 = self.minmax_param(params[id_2])

        Sigma_1 = np.identity(self.n) * np.square(self.minmax_sigma(self.sigma[id_1]))
        Sigma_2 = np.identity(self.n) * np.square(self.minmax_sigma(self.sigma[id_2]))
        Sigma = (Sigma_1 + Sigma_2) / 2

        return (param_1 - param_2).T @ np.linalg.pinv(Sigma) @ (param_1 - param_2) / 16 \
                + np.log(np.linalg.det(Sigma) / np.sqrt(np.linalg.det(Sigma_1) * np.linalg.det(Sigma_2))) / 2
    
    def Kullback_Leibler_divergence(self, params, id_1, id_2) -> np.ndarray:
        param_1 = self.minmax_param(params[id_1])
        param_2 = self.minmax_param(params[id_2])

        Sigma_1 = np.identity(self.n) * np.square(self.minmax_sigma(self.sigma[id_1]))
        Sigma_2 = np.identity(self.n) * np.square(self.minmax_sigma(self.sigma[id_2]))

        Sigma = np.linalg.pinv(Sigma_2) @ Sigma_1

        return (param_1 - param_2).T @ np.linalg.pinv(Sigma_2) @ (param_1 - param_2) \
                - np.log(np.linalg.det(Sigma)) + np.trace(Sigma) - self.n

    def correlation(self, params) -> np.ndarray:
        corr = np.identity(self.num_pop)

        for i in np.arange(self.num_pop):
            for j in np.arange(self.num_pop):
                if i != j:
                    corr[i][j] = self.Bhattacharyya_Distance(params, i, j)

        return np.min(corr, axis = 1).reshape(-1, 1)

    def replace_father(self) -> None:
        correlation_father = self.correlation(self.params_father)
        correlation_child = self.correlation(self.params_child)
        corr = correlation_child / (correlation_child + correlation_father)
        reward = (self.reward_child - self.best_score) / (self.reward_child + self.reward_father - 2 * self.best_score)

        update = (reward / corr < self.lambda_t).flatten()
        self.params_father[update] = self.params_child[update]
        self.reward_father[update] = self.reward_child[update]
        self.update_count[update] += 1

    def update_sigma(self) -> None:
        update = (self.update_count / self.epoch).flatten()
        
        self.sigma[update < 0.2] /= self.r
        self.sigma[update > 0.2] *= self.r

        self.sigma_init = self.sigma_init * 0.95
        self.sigma = np.minimum(self.sigma, self.sigma_init)

    def log(self) -> None:
        with open('log/log_' + str(self.id) + '.txt', 'a') as f:
            f.write("iter: " + str(self.iter) + " best score: " + str(self.best_score) + " best param: " + str(self.best_param) + "\n")
    
    def step(self) -> None:
        self.update_lambda()
        self.generate_child()
        with Pool(self.num_pop) as p:
            id = np.arange(self.num_pop)
            self.reward_child = np.array([p.map(self.f, self.params_child)]).reshape(-1, 1)
        self.update_best()
        self.replace_father()

        if self.iter % self.epoch == 0:
            self.update_sigma()
            self.update_count = np.zeros((self.num_pop, 1))

    def run(self) -> None:
        with Pool(self.num_pop) as p:
            id = np.arange(self.num_pop)
            self.reward_father = np.array([p.map(self.f, self.params_father)]).reshape(-1, 1)

        self.best_score = np.min(self.reward_father)
        self.best_param = self.params_father[np.argmin(self.reward_father)]

        self.log()
        self.iter += 1
        
        while(self.iter < self.max_iter):
            self.step()

            self.log()
            self.iter += 1