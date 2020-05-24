import numpy as np
import copy as cp
from numpy.random import normal


def CMSA(f, sub_popu, sub_size, dim, improve_generation, lb, ub, total_size, local_optimal):
    half_sub_size = max(1, int(sub_size / 2))
    eel = get_bandwidth(dim, lb, ub, total_size)
    sub_popu = sorted(sub_popu, key=lambda x: x[1], reverse=True)
    best_so_far_indiv = cp.deepcopy(sub_popu[0])
    tau, tau_c, weights, covariance, cholesky = init_para(sub_popu, len(sub_popu), dim, lb, ub, total_size)
    mean = get_weighted_mean(sub_popu, dim, weights)
    sigma, sigma_l = 1.0, 1.0
    eval_times = 0
    covariance, cholesky = update_covariance(covariance, sub_popu, len(sub_popu), dim, weights, tau_c)
    new_popu = []
    for i in range(sub_size):
        sigma_l = sigma * np.exp(tau * normal())
        s_l = cholesky.dot(normal(size=dim))
        z_l = sigma_l * s_l
        y_l = mean + z_l
        for index in range(dim):
            if y_l[index] < lb[index]:
                y_l[index] = lb[index]
            if y_l[index] > ub[index]:
                y_l[index] = ub[index]
        f_l = f.evaluate(y_l)
        new_popu.append([y_l, f_l, sigma_l, s_l])
        eval_times += 1

    # sub_popu = sub_popu[:max(1, int(len(sub_popu) * 0.3))]
    # sub_popu.extend(new_popu)
    sub_popu = sorted(new_popu, key=lambda x: x[1], reverse=True)
    sub_popu = sub_popu[:half_sub_size]
    if sub_popu[0][1] > best_so_far_indiv[1]:
        best_so_far_indiv = cp.deepcopy(sub_popu[0])

    ## -------------------------------------------------------------

    # tau, tau_c, weights, covariance, cholesky = init_para(sub_popu, sub_size, dim, lb, ub, total_size)
    generation = 0
    tau, tau_c, weights, _, _ = init_para(sub_popu, half_sub_size, dim, lb, ub, total_size)
    without_improve = 0
    while without_improve < improve_generation:
        generation += 1
        mean = get_weighted_mean(sub_popu, dim, weights)
        sigma = update_sigma(sigma, half_sub_size, weights, sub_popu)
        covariance, cholesky = update_covariance(covariance, sub_popu, half_sub_size, dim, weights, tau_c)
        new_popu = []
        for i in range(sub_size):
            sigma_l = sigma * np.exp(tau * normal())
            s_l = cholesky.dot(normal(size=dim))
            z_l = sigma_l * s_l
            y_l = mean + z_l
            for index in range(dim):
                if y_l[index] < lb[index]:
                    y_l[index] = lb[index]
                if y_l[index] > ub[index]:
                    y_l[index] = ub[index]
            f_l = f.evaluate(y_l)
            new_popu.append([y_l, f_l, sigma_l, s_l])
            eval_times += 1

        # sub_popu = sub_popu[:max(1, int(len(sub_popu) * 0.3))]
        # sub_popu.extend(new_popu)
        sub_popu = sorted(new_popu, key=lambda x: x[1], reverse=True)
        sub_popu = sub_popu[:half_sub_size]
        # sub_popu = sub_popu[:sub_size]
        # sub_popu = sorted(new_popu, key=lambda x: x[1], reverse=True)

        # sub_popu = sorted(new_popu, key=lambda x: x[1], reverse=True)
        if sub_popu[0][1] > best_so_far_indiv[1]:
            if sub_popu[0][1] > best_so_far_indiv[1] + 0.00001:
                without_improve = 0
            best_so_far_indiv = cp.deepcopy(sub_popu[0])
        else:
            without_improve += 1

        sub_popu_best = cp.deepcopy(sub_popu[0])
        if generation % 5 == 0 and len(local_optimal) != 0:
            dis = np.zeros(len(local_optimal), dtype=float)
            for index in range(len(local_optimal)):
                dis[index] = np.sum(np.abs(local_optimal[index][0] - sub_popu_best[0]))
            nearest_index = dis.argmin()

            test_num = min(10, 1 + int(dis[nearest_index] / eel))
            same_niching, e = hill_valley_test(f, local_optimal[nearest_index], sub_popu_best, test_num)
            eval_times += e
            if same_niching:
                break

    sub_popu.append(best_so_far_indiv)
    sub_popu = sorted(sub_popu, key=lambda x: x[1], reverse=True)
    return sub_popu, eval_times


def hill_valley_test(f, indiv1, indiv2, test_num):
    min_fitness = min(indiv1[1], indiv2[1])
    test_num += 1
    evaluate_times = 0
    for i in range(1, test_num):
        evaluate_times += 1
        temp_indiv = indiv1[0] + i / test_num * (indiv2[0] - indiv1[0])
        temp_fitness = f.evaluate(temp_indiv)
        if temp_fitness < min_fitness:
            return False, evaluate_times

    return True, evaluate_times


def get_weights(sub_size):
    weights = np.zeros(sub_size, dtype=float)
    sum_weight = 0.0
    for i in range(sub_size):
        weights[i] = np.log(sub_size + 1) - np.log(i + 1)
        sum_weight += weights[i]

    for i in range(sub_size):
        weights[i] = weights[i] / sum_weight

    return weights


def get_bandwidth(dim, lb, ub, total_size):
    scaled_search_volume = 1.0
    dim = 1.0 / dim
    for low, up in zip(lb, ub):
        scaled_search_volume *= np.power(up - low, dim)

    return scaled_search_volume * np.power(total_size, -dim)


def get_mean(sub_popu, dim):
    mean = np.zeros(dim, dtype=float)
    for indiv in sub_popu:
        mean += indiv[0]
    mean /= len(sub_popu)

    return mean


def get_weighted_mean(sub_popu, dim, weights):
    mean = np.zeros(dim, dtype=float)
    for i in range(len(sub_popu)):
        mean += sub_popu[i][0] * weights[i]
    # mean /= len(sub_popu)

    return mean


def init_strategy_parameters(dim, sub_size):
    t_cov_coeff = 1.0
    tau = 1.0 / np.sqrt(2.0 * dim)
    tau_c = 1 + dim * (dim + 1.0) / sub_size * t_cov_coeff
    weights = get_weights(sub_size)
    return tau, tau_c, weights


def init_para(sub_popu, sub_size, dim, lb, ub, total_size):
    t_cov_coeff = 1.0
    tau = 1.0 / np.sqrt(2.0 * dim)
    tau_c = 1 + dim * (dim + 1.0) / sub_size * t_cov_coeff
    weights = get_weights(sub_size)

    if len(sub_popu) == 1:
        bandwidth = get_bandwidth(dim, lb, ub, total_size)
        # bandwidth = get_bandwidth(dim, lb, ub, sub_size)
        covariance = np.eye(dim) * bandwidth * 0.01
        cholesky = np.eye(dim) * np.sqrt(bandwidth * 0.01)
    else:
        mean = get_mean(sub_popu, dim)
        covariance = covariance_univariate(sub_popu, dim, mean)
        cholesky = cholesky_decomposition_univariate(covariance, dim)

    return tau, tau_c, weights, covariance, cholesky


# population covariance
def covariance_univariate(sub_popu, dim, mean):
    covariance = np.zeros((dim, dim), dtype=float)
    size = len(sub_popu)
    for i in range(dim):
        for k in range(size):
            covariance[i][i] += (sub_popu[k][0][i] - mean[i]) * (sub_popu[k][0][i] - mean[i])
        covariance[i][i] /= size

    return covariance


def cholesky_decomposition_univariate(cov, dim):
    cholesky = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        cholesky[i][i] = np.sqrt(cov[i][i])

    return cholesky


def update_sigma(sigma, sub_size, weights, sub_popu):
    sum_logsigma = weighted_sum_logsigma = 0.0
    for i in range(sub_size):
        weighted_sum_logsigma += weights[i] * np.log(sub_popu[i][2])
        sum_logsigma += np.log(sub_popu[i][2])

    sigma = sigma * np.exp(weighted_sum_logsigma) / np.exp(sum_logsigma / sub_size)

    return sigma


def cholesky_decomposition(cov, dim):
    eigenvalues = np.linalg.eigvals(cov)
    if np.all(eigenvalues > 0):
        try:
            return np.linalg.cholesky(cov)
        except:
            pass

    cholesky = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        cholesky[i][i] = np.sqrt(cov[i][i])
    return cholesky


def update_covariance(covariance, sub_popu, sub_size, dim, weights, tau_c):
    sst = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(i, dim):
            for k in range(sub_size):
                sst[i][j] += weights[k] * sub_popu[k][3][i] * sub_popu[k][3][j]
            sst[j][i] = sst[i][j]

    new_covariance = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(i, dim):
            new_covariance[i][j] = (1.0 - 1.0 / tau_c) * covariance[i][j] + (1.0 / tau_c) * sst[i][j]
            new_covariance[j][i] = new_covariance[i][j]

    cholesky = cholesky_decomposition(new_covariance, dim)
    return new_covariance, cholesky
