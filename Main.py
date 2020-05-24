import copy as cp

import numpy as np

from CMSA import CMSA
from cec2013.cec2013 import *


def init_popu(size, dim, lb, ub):
    # Create population within bounds
    popu = []
    for i in range(size):
        popu.append([lb + (ub - lb) * np.random.rand(dim), 0.0, 1.0, [0.0 for _ in range(dim)]])

    return popu


def merge_popu(p1, p2):
    popu = cp.deepcopy(p1)
    popu.extend(p2)

    return popu


def get_basic_para(f):
    dim = f.get_dimension()
    size = dim * 16
    ub = np.zeros(dim)
    lb = np.zeros(dim)
    # Get lower, upper bounds
    for i in range(dim):
        ub[i] = f.get_ubound(i)
        lb[i] = f.get_lbound(i)

    return size, dim, ub, lb, f.get_maxfes(), f.get_rho()


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


def get_bandwidth(dim, lb, ub, total_size):
    scaled_search_volume = 1.0
    dim = 1.0 / dim
    for low, up in zip(lb, ub):
        scaled_search_volume *= np.power(up - low, dim)

    return scaled_search_volume * np.power(total_size, -dim)


def hill_valley_clustering(popu, f, size, dim, lb, ub, eel):
    cluster_num = 0
    cluster = np.zeros(size, dtype=int)
    popu = sorted(popu, key=lambda x: x[1], reverse=True)

    total_eval_times = 0
    for i in range(1, size):
        dis = np.zeros(i, dtype=float)
        for j in range(i):
            # dis[j] = np.sum(np.square(popu[i][0] - popu[j][0]))
            dis[j] = np.sum(np.abs(popu[i][0] - popu[j][0]))
        dis_arg_sort = np.argsort(dis)

        checked_cluster = set()
        new_cluster = True
        for j in range(min(i, dim + 1)):
            nearest_index = dis_arg_sort[j]
            if nearest_index not in checked_cluster:
                nearest_dis = np.sum(np.square(popu[i][0] - popu[nearest_index][0]))
                # test_num = min(10, 1 + int(dis[nearest_index] / eel))
                test_num = min(10, 1 + int(nearest_dis / eel))
                same_niching, eval_times = hill_valley_test(f, popu[i], popu[nearest_index], test_num)
                total_eval_times += eval_times
                if same_niching:
                    cluster[i] = cluster[nearest_index]
                    new_cluster = False
                    break

                checked_cluster.add(nearest_index)

        if new_cluster:
            cluster_num += 1
            cluster[i] = cluster_num

    return popu, cluster, cluster_num, total_eval_times


def main():
    TOL = 0.001
    for problem_index in range(8, 9):
        for run in range(1):
            print("=" * 7)
            # Create function
            f = CEC2013(problem_index)
            size, dim, ub, lb, max_eval_times, rho = get_basic_para(f)
            cur_eval_times = 0
            elitist_archive = []
            # sub_size = int(10 * np.sqrt(dim))
            sub_size = max(5, int(3 * np.sqrt(dim)) + 1)

            while True:
                local_optimal = cp.deepcopy(elitist_archive)
                if cur_eval_times > max_eval_times:
                    break
                eel = get_bandwidth(dim, lb, ub, size)
                double_size = True
                improve_generation = 10 + int(30 * dim / sub_size)

                population = init_popu(size, dim, lb, ub)
                for indiv in population:
                    indiv[1] = f.evaluate(indiv[0])
                cur_eval_times += len(population)
                population = merge_popu(elitist_archive, population)
                population = sorted(population, key=lambda x: x[1], reverse=True)
                population = population[:size]

                population, cluster, cluster_num, eval_times = hill_valley_clustering(population, f, size, dim, lb, ub,
                                                                                      eel)

                cur_eval_times += eval_times
                if cur_eval_times > max_eval_times:
                    break

                for c in range(cluster_num + 1):
                    sub_popu = []
                    for i in range(size):
                        if cluster[i] == c:
                            sub_popu.append(cp.deepcopy(population[i]))
                    # sub_popu, eval_times = GA(f, sub_popu, sub_size, dim, improve_generation, lb, ub)
                    # sub_popu, eval_times = CMSA(f, sub_popu, sub_size, dim, improve_generation, lb, ub, size)
                    # sub_popu, eval_times = CMSA_order(f, sub_popu, sub_size, dim, improve_generation, lb, ub, size)
                    sub_popu, eval_times = CMSA(f, sub_popu, sub_size, dim, improve_generation, lb, ub, size,
                                                local_optimal)
                    sub_popu = sorted(sub_popu, key=lambda x: x[1], reverse=True)
                    cur_eval_times += eval_times
                    if cur_eval_times > max_eval_times:
                        break
                    sub_popu_best = cp.deepcopy(sub_popu[0])
                    local_optimal.append(cp.deepcopy(sub_popu_best))

                    if len(elitist_archive) != 0:
                        largest_fitness = elitist_archive[0][1]
                    else:
                        largest_fitness = 0
                    for elite in elitist_archive:
                        if elite[1] > largest_fitness:
                            largest_fitness = elite[1]

                    if len(elitist_archive) == 0 or sub_popu_best[1] > largest_fitness + TOL:
                        elitist_archive = [sub_popu_best]
                        continue

                    if sub_popu_best[1] > largest_fitness - TOL:
                        dis = np.zeros(len(elitist_archive), dtype=float)
                        for index in range(len(elitist_archive)):
                            dis[index] = np.sum(np.abs(elitist_archive[index][0] - sub_popu_best[0]))
                        nearest_index = dis.argmin()

                        test_num = min(10, 1 + int(dis[nearest_index] / eel))
                        same_niching, eval_times = hill_valley_test(f, elitist_archive[nearest_index], sub_popu_best,
                                                                    test_num)
                        cur_eval_times += eval_times
                        if cur_eval_times > max_eval_times:
                            break

                        if same_niching:
                            if sub_popu_best[1] > elitist_archive[nearest_index][1]:
                                elitist_archive[nearest_index] = sub_popu_best
                            continue
                        elitist_archive.append(sub_popu_best)
                        double_size = False

                if double_size:
                    size = min(dim * 1000, size * 2)
                    # size = size * 2
                    sub_size = int(sub_size * 1.2)
                print(cur_eval_times, len(local_optimal), len(elitist_archive), np.array(elitist_archive)[:, 1])

            rst = []
            for elite in elitist_archive:
                rst.append(elite[0])

            count, seeds = how_many_goptima(np.array(rst), f, 0.1)
            print(problem_index, run, count)
            count, seeds = how_many_goptima(np.array(rst), f, 0.01)
            print(problem_index, run, count)
            count, seeds = how_many_goptima(np.array(rst), f, 0.001)
            print(problem_index, run, count)
            count, seeds = how_many_goptima(np.array(rst), f, 0.0001)
            print(problem_index, run, count)
            count, seeds = how_many_goptima(np.array(rst), f, 0.00001)
            # print("Global optimizers:", seeds)
            print(problem_index, run, count)


main()
