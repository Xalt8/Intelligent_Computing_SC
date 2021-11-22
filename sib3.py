import numpy as np
from dataclasses import dataclass
from numba import njit 
import time
from pso_ex import PSO, poss_val, random_val, feasible_vec, non_zero_inds, calculate_profit, experiment, split_particles_list
import pandas as pd

@njit
def sib_mix(vec:np.ndarray, better_vec: np.ndarray)-> np.ndarray:
    ''' Returns an array (vec) that contains a percentage of values 
        from another array (better_vec)
        Parameters:
        ===========
        percent_to_replace: percentage of values to exchange
        qb: number of values to exchange
        diff_index: the index position with absolute different between 
                    the 2 vectors in sorted order -> biggest difference first       
        '''
    mix_vec = vec.copy()
    percent_to_replace = 0.10
    qb = int(np.ceil(np.where(vec!=better_vec,1,0).sum() * percent_to_replace))
    diff_index = np.argsort(np.absolute(better_vec - mix_vec))[::-1]
    
    if qb < diff_index.size:
        for ind in diff_index:
            if qb == 0:
                break
            elif poss_val(index=ind, val=better_vec[ind], vec=mix_vec):
                mix_vec[ind]=better_vec[ind]
                qb -= 1
            else:
                continue
    else: 
        # qb > diff_index.size:
        for ind in diff_index:
            if poss_val(index=ind, val=better_vec[ind], vec=mix_vec):
                mix_vec[ind]=better_vec[ind]
            else:
                continue
    
    assert feasible_vec(mix_vec), 'sib_mix() returned an unfeasible vector' 
    return mix_vec


def random_jump(vec: np.ndarray) -> np.ndarray:
    ''' Replaces a percentage of values from a vector with random values that
        meet demand and supply constraints
        non_zero_inds: index positions that have demand '''

    global non_zero_inds
    new_vec = vec.copy()
    percent_to_replace = 0.40
    nos_to_replace = int(percent_to_replace * non_zero_inds.size)
    inds = np.random.choice(non_zero_inds, nos_to_replace, replace=False)
    for i in inds:
        new_vec[i] = random_val(vec=new_vec, index=i)
    
    assert feasible_vec(new_vec), 'random_jump() returned an unfeasible vector'
    return new_vec



@dataclass
class SIB3(PSO):

    def mix(self):
        for particle in self.particles:
            particle['mixwGB_pos'] = sib_mix(particle['position'], self.gbest_pos)
            particle['mixwGB_val'] = calculate_profit(particle['mixwGB_pos'])
            particle['mixwLB_pos'] = sib_mix(particle['position'], particle['lbest_pos'])
            particle['mixwLB_val'] = calculate_profit(particle['mixwLB_pos'])

    def move(self):
        for particle in self.particles:
            if particle['mixwLB_val'] >= particle['mixwGB_val'] >= particle['pbest_val']:
                particle['position'] = particle['mixwLB_pos']
            elif particle['mixwGB_val'] >= particle['mixwLB_val']>= particle['pbest_val']:
                particle['position'] = particle['mixwGB_pos']
            else:
                particle['position'] = random_jump(particle['position'])


def optimize(init_pos:np.ndarray) -> tuple[list[np.ndarray], float]:

    optimize.counter += 1

    start = time.perf_counter()

    iterations = 50

    gbest_val_list  = []
    gbest_pos_list  = []
    
    swarm = SIB3()
    swarm.initialise_with_particle_list(init_pos)
    swarm.pick_informants_ring_topology()
    
    for i in range(iterations):
        swarm.calculate_fitness()
        swarm.set_pbest()
        swarm.set_lbest()
        swarm.set_gbest()
        swarm.mix()
        swarm.move()
        
        print(f"Run:{optimize.counter}, Iteration: {i}, gbest_val: {swarm.gbest_val:.2f}")    

        gbest_val_list.append(round(swarm.gbest_val, 2))
        if i == iterations-1: # Last iteration
            gbest_pos_list.append(swarm.gbest_pos)
            if feasible_vec(swarm.gbest_pos):
                print("Constraints met!")

    end = time.perf_counter()
    total_time = end-start

    return gbest_val_list, total_time


if __name__ == '__main__':

    optimize.counter = 0
    experiment(optimise_func=optimize, split_particles_list=split_particles_list, experiment_name='sib3_ex_10')
