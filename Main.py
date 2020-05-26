import copy as cp
import time

from CMSA import CMSA
from cec2013.cec2013 import *


def init_popu(size, dim, lb, ub):
    # Create population within bounds
    popu = []
    for i in range(size):
        popu.append([lb + (ub - lb) * np.random.rand(dim), 0.0, 1.0, [0.0 for _ in range(dim)]])

    return popu


def merge_popu(p1, p2):
    p2 = cp.deepcopy(p2)
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

    return size, dim, ub, lb, f.get_maxfes()


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


def update_quality_index(restart_times, low_s, high_s, low_quality_index, high_quality_index):
    if restart_times % 2 == 0 and restart_times > 0:
        if low_s > high_s:
            low_quality_index = max(1, low_quality_index // 2)
            high_quality_index = (low_quality_index + high_quality_index) // 2
        elif low_s < high_s:
            high_quality_index = min(100, high_quality_index * 2)
            low_quality_index = (low_quality_index + high_quality_index) // 2

    return low_quality_index, high_quality_index


def write2file(problem_index, run, elitist_archive, fes, eva_time):
    temp = zip(elitist_archive, fes, eva_time)
    temp = sorted(temp, key=lambda x: (x[1], x[2]))
    with open('./rst/problem' + '%03d' % problem_index + 'run%03d' % run + '.dat', 'w') as file:
        for line in temp:
            for x in line[0][0]:
                file.write(str(x) + ' ')
            file.write('= ' + str(line[0][1][0]) + ' @ ')
            file.write(str(line[1]) + ' ')
            file.write(str(line[2]) + ' 1\n')


def write_acc(f, problem_index, elitist_archive):
    rst = []
    for elite in elitist_archive:
        rst.append(elite[0])

    num_opt = f.get_info()['nogoptima']
    with open(str(problem_index) + '_accuracy.txt', 'a') as file:
        count, seeds = how_many_goptima(np.array(rst), f, 0.1)
        file.write(str(count / num_opt) + ' ')
        count, seeds = how_many_goptima(np.array(rst), f, 0.01)
        file.write(str(count / num_opt) + ' ')
        count, seeds = how_many_goptima(np.array(rst), f, 0.001)
        file.write(str(count / num_opt) + ' ')
        count, seeds = how_many_goptima(np.array(rst), f, 0.0001)
        file.write(str(count / num_opt) + ' ')
        count, seeds = how_many_goptima(np.array(rst), f, 0.00001)
        file.write(str(count / num_opt) + '\n')


def main():
    TOL = 0.00001
    for problem_index in range(2, 4):
        for run in range(1, 51):
            print(problem_index, run)
            start = time.time()
            np.random.seed(problem_index * 1000 + run)
            # Create function
            f = CEC2013(problem_index)
            size, dim, ub, lb, max_eval_times = get_basic_para(f)
            cur_eval_times = 0
            elitist_archive, fes, eva_time = [], [], []
            sub_size = max(5, int(3 * np.sqrt(dim)) + 1)
            restart_times = -1
            low_quality_index, high_quality_index = 2, 50
            low_quality_solution, high_quality_solution = 0, 0

            while True:
                # print(low_quality_index, high_quality_index)
                restart_times += 1
                low_quality_index, high_quality_index = update_quality_index(restart_times, low_quality_solution,
                                                                             high_quality_solution, low_quality_index,
                                                                             high_quality_index)

                local_optimal = cp.deepcopy(elitist_archive)
                if cur_eval_times > max_eval_times:
                    break
                eel = get_bandwidth(dim, lb, ub, size)
                double_size = True
                improve_generation = 10 + int(30 * dim / sub_size)

                if restart_times % 2 == 0:
                    population = init_popu(size * low_quality_index, dim, lb, ub)
                else:
                    population = init_popu(size * high_quality_index, dim, lb, ub)

                for indiv in population:
                    indiv[1] = f.evaluate(indiv[0])
                cur_eval_times += len(population)
                population = sorted(population, key=lambda x: x[1], reverse=True)
                population = population[:size]
                population = merge_popu(elitist_archive, population)
                population = sorted(population, key=lambda x: x[1], reverse=True)

                population, cluster, cluster_num, eval_times = hill_valley_clustering(population, f, len(population),
                                                                                      dim, lb, ub,
                                                                                      eel)
                cur_eval_times += eval_times
                if cur_eval_times > max_eval_times:
                    break

                for c in range(cluster_num + 1):
                    sub_popu = []
                    for i in range(len(population)):
                        if cluster[i] == c:
                            sub_popu.append(cp.deepcopy(population[i]))
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
                        largest_fitness = -100000000
                    for elite in elitist_archive:
                        if elite[1] > largest_fitness:
                            largest_fitness = elite[1]

                    if len(elitist_archive) == 0 or sub_popu_best[1] > largest_fitness + TOL:
                        elitist_archive = [sub_popu_best]
                        fes = [cur_eval_times]
                        eva_time = [(time.time() - start) * 1000]
                        if restart_times % 2 == 0:
                            low_quality_solution = 1
                            high_quality_solution = 0
                        else:
                            low_quality_solution = 0
                            high_quality_solution = 1

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
                                fes[nearest_index] = cur_eval_times
                                eva_time[nearest_index] = (time.time() - start) * 1000
                                if restart_times % 2 == 0:
                                    low_quality_solution += 1
                                else:
                                    high_quality_solution += 1
                            continue
                        elitist_archive.append(sub_popu_best)
                        fes.append(cur_eval_times)
                        eva_time.append((time.time() - start) * 1000)
                        if restart_times % 2 == 0:
                            low_quality_solution += 1
                        else:
                            high_quality_solution += 1
                        double_size = False

                if double_size:
                    size = min(dim * 1000, size * 2)
                    sub_size = int(sub_size * 1.2)

            write_acc(f, problem_index, elitist_archive)
            write2file(problem_index, run, elitist_archive, fes, eva_time)


main()
