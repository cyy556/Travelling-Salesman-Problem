#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:04:39 2021

@author: julia
"""
#create graph 
import random
def Graph(n):
    matrix = [([0] * n) for i in range(n)] 
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0
            else:
                matrix[i][j] = random.randrange(1, 30)
    return matrix

#dynamic programming
import copy
def TSP_DP(matrix, n):
    g = {}
    p = []

    for x in range(1, n):
        g[x + 1, ()] = matrix[x][0]

    weight = get_minimum(1, (range(2, n + 1)), g, p)
    return weight

def get_minimum(k, a, g, p):
    if (k, a) in g:
        # Already calculated Set g[%d, (%s)]=%d' % (k, str(a), g[k, a]))
        return g[k, a]

    values = []
    all_min = []
    for j in a:
        set_a = copy.deepcopy(list(a))
        set_a.remove(j)
        all_min.append([j, tuple(set_a)])
        result = get_minimum(j, tuple(set_a), g, p)
        values.append(matrix[k - 1][j - 1] + result)

    # get minimun value from set as optimal solution for
    g[k, a] = min(values)
    p.append(((k, a), all_min[values.index(g[k, a])]))

    return g[k, a]

#ant algorithm
#https://medium.com/qiubingcheng/以python實作蟻群最佳化演算法-ant-colony-optimization-aco-並解決tsp問題-上-b8c1a345c5a1
import numpy as np
import sys

node = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

class TSPProblem:
    def __init__(self, coordinate, cities_name):
        self.coordinate = coordinate
        self.cities_name = cities_name

    def get_distance(self, arr1, arr2):
        # Euclidean distance
        # return np.sqrt(np.power(arr1 - arr2, 2).sum())
        # definded distance
        return self.coordinate[arr1][arr2]

    def compute_objective_value(self, cities_id):
        total_distance = 0
        for i in range(len(cities_id)):
            city1 = cities_id[i]
            city2 = cities_id[i + 1] if i < len(cities_id) - 1 else cities_id[0]
            # total_distance += self.get_distance(self.coordinate[city1], self.coordinate[city2])
            total_distance += self.get_distance(city1, city2)
        return total_distance

    def to_cities_name(self, cities_id):
        return [self.cities_name[i] for i in cities_id]
    
class AntSystem:
    def __init__(self, pop_size, coordinate, pheromone_drop_amount, evaporate_rate,
                 pheromone_factor, heuristic_factor,
                 get_distance, compute_objective_value):

        self.num_ants = pop_size
        self.coordinate = coordinate
        self.num_cities = len(coordinate)
        self.get_distance = get_distance
        self.compute_objective_value = compute_objective_value
        self.pheromone_drop_amount = pheromone_drop_amount
        self.evaporate_rate = evaporate_rate
        self.pheromone_factor = pheromone_factor
        self.visibility_factor = heuristic_factor

    def initialize(self):
        self.one_solution = np.arange(self.num_cities, dtype=int)
        self.solutions = np.zeros((self.num_ants, self.num_cities), dtype=int)
        for i in range(self.num_ants):
            for c in range(self.num_cities):
                self.solutions[i][c] = c

        self.objective_value = np.zeros(self.num_ants)
        self.best_solution = np.zeros(self.num_cities, dtype=int)
        self.best_objective_value = sys.float_info.max

        self.visibility = np.zeros((self.num_cities, self.num_cities))
        self.pheromone_map = np.ones((self.num_cities, self.num_cities))

        # heuristic_values
        for from_ in range(self.num_cities):
            for to in range(self.num_cities):
                if (from_ == to): continue
                # distance = self.get_distance(self.coordinate[from_], self.coordinate[to])
                distance = self.get_distance(from_, to)
                self.visibility[from_][to] = 1 / distance

    def do_roulette_wheel_selection(self, fitness_list):
        kk = 0
        for i in range(len(fitness_list) - 1):
            kk += fitness_list[i]
        transition_probability = [fitness / kk for fitness in fitness_list]

        rand = random.random()
        sum_prob = 0
        for i, prob in enumerate(transition_probability):
            sum_prob += prob
            if (sum_prob >= rand):
                return i

    def update_pheromone(self):
        # evaporate hormones all the path
        self.pheromone_map *= (1 - self.evaporate_rate)

        # Add hormones to the path of the ants
        for solution in self.solutions:
            for j in range(self.num_cities):
                city1 = solution[j]
                city2 = solution[j + 1] if j < self.num_cities - 1 else solution[0]
                self.pheromone_map[city1, city2] += self.pheromone_drop_amount

    def _an_ant_construct_its_solution(self):
        candidates = [i for i in range(self.num_cities)]
        # random choose city as first city
        current_city_id = random.choice(candidates)
        self.one_solution[0] = current_city_id
        candidates.remove(current_city_id)

        # select best from candiate
        for t in range(1, self.num_cities - 1):
            # best
            fitness_list = []
            for city_id in candidates:
                fitness = pow(self.pheromone_map[current_city_id][city_id], self.pheromone_factor) * \
                          pow(self.visibility[current_city_id][city_id], self.visibility_factor)
                fitness_list.append(fitness)

            next_city_id = candidates[self.do_roulette_wheel_selection(fitness_list)]
            candidates.remove(next_city_id)
            self.one_solution[t] = next_city_id

            current_city_id = next_city_id
        self.one_solution[-1] = candidates.pop()

    def each_ant_construct_its_solution(self):
        for i in range(self.num_ants):
            self._an_ant_construct_its_solution()
            for c in range(self.num_cities):
                self.solutions[i][c] = self.one_solution[c]

            self.objective_value[i] = self.compute_objective_value(self.solutions[i])

    def update_best_solution(self):
        for i, val in enumerate(self.objective_value):
            if (val < self.best_objective_value):
                for n in range(self.num_cities):
                    self.best_solution[n] = self.solutions[i][n]

                self.best_objective_value = val

#main
import time
import csv
count = 1

for V in range(4, 21):
    while count != 6:
        print("test ", count)
        print("vertex number: ", V)
        matrix = Graph(V)
    
        #run DP
        start = time.time()
        r1 = TSP_DP(matrix, V)
        print("DP Weight:", r1)
        end = time.time()
        t1 = end - start
        print("DP_time:", t1)
    
        #run AS
        problem = TSPProblem(matrix, node)
    
        pop_size = 20
        pheromone_drop_amount = 0.001
        evaporate_rate = 0.1
        pheromone_factor = 1
        heuristic_factor = 3
    
        solver = AntSystem(pop_size, matrix, pheromone_drop_amount, evaporate_rate,
                           pheromone_factor, heuristic_factor,
                           problem.get_distance, problem.compute_objective_value)
    
        solver.initialize()
        start = time.time()
        solver.each_ant_construct_its_solution()
        solver.update_pheromone()
        solver.update_best_solution()
        r2 = int(solver.best_objective_value)
        print("AS _weight:", r2)
        end = time.time()
        t2 = end - start
        print("AS_time:", t2)
        
        #calculate error
        error = (r2 - r1)/r1
        print("error:", error)
        
        print("                                                 ")
    
        with open('TSP_data(DP, AS).csv', 'a+', newline='')as csvFile:        
            csvWriter = csv.writer(csvFile)
            #csvWriter.writerow(['vertex number', 'dp weight','na weight','dp time', 'na time', 'error'])
            csvWriter.writerow([str(V), str(r1), str(r2), str(t1), str(t2), str(error)])    
        csvFile.close()
        count += 1
    count = 1
    print("-------------------------------------------------")
    
