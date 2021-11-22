import numpy as np
from dataclasses import dataclass
import time
# from pso import random_val, poss_val, feasible_vec, calculate_profit, plot_results, experiment, split_particles_list, rs
from sib3 import SIB3, random_jump
from pso_ex import PSO, feasible_vec, poss_val, random_val, plot_results, calculate_profit, non_zero_inds, experiment, split_particles_list
from numba import njit 


# @njit
def get_r1r2(index:int, vec:np.ndarray,  better_vec:np.ndarray) -> int:
    ''' Takes 2 values and returns a random number 
        between those values if it meets constraints (r1)
        otherwise returns any random value that meets constraints (r2)'''
    min_val = np.minimum(better_vec[index], vec[index])
    max_val = np.maximum(better_vec[index], vec[index])
    # if max_val > min_val > 0:
    r1 = np.random.randint(np.maximum(min_val, 0), max_val)
    if poss_val(index=index, val=r1, vec=vec):
        return r1
    else:
        r2 = random_val(vec=vec, index=index)
        if poss_val(index=index, val=r2, vec=vec):
            return r2


@njit
def sibv_mix(vec: np.ndarray, better_vec: np.ndarray) -> np.ndarray:
    ''' Returns an array (vec) that contains a percentage of values 
        from another array (better_vec)
        Parameters:
        ===========
        percent_to_replace: The amount of values in percent to exchange
        qb: the total number of values to exchange
        diff_index: the index position that absolute different between 
                    the 2 vectors in sorted order -> biggest difference first       
        '''
    new_pos = vec.copy()
    percent_to_replace = 0.40
    qb = int(np.ceil(np.where(vec!=better_vec,1,0).sum() * percent_to_replace))
    diff_index = np.argsort(np.absolute(better_vec - vec))[::-1]
               
    if qb < diff_index.size:
        for ind in diff_index:
            while qb > 0:
                r1r2 = get_r1r2(index=ind, vec=vec, better_vec=better_vec)
                new_pos[ind]=r1r2
                qb -= 1
    else: 
        # qb > diff_index.size:
        for ind in diff_index:
            r1r2 = get_r1r2(index=ind, vec=vec, better_vec=better_vec)
            new_pos[ind]=r1r2
        
    assert feasible_vec(new_pos), 'sibv_mix() returned unfeasible vector'  
    return new_pos


    
@dataclass
class SIBV3(SIB3):
    def mix(self):
        for particle in self.particles:
            # mixwGB
            particle['mixwGB_pos'] = sibv_mix(particle['position'], self.gbest_pos)
            particle['mixwGB_val'] = calculate_profit(particle['mixwGB_pos'])
            # mixwLB
            particle['mixwLB_pos'] = sibv_mix(particle['position'], particle['lbest_pos'])
            particle['mixwLB_val'] = calculate_profit(particle['mixwLB_pos'])
            #mixwPB
            particle['mixwPB_pos'] = sibv_mix(particle['position'], particle['pbest_pos'])
            particle['mixwPB_val'] = calculate_profit(particle['mixwPB_pos'])


    def move(self):
        for particle in self.particles:
            # mixwLB is best
            if particle['mixwLB_val'] >= particle['mixwGB_val'] >= particle['mixwPB_val']:
                particle['position'] = particle['mixwLB_pos']
            # mixwGB is best
            elif particle['mixwGB_val'] >= particle['mixwLB_val']>= particle['mixwPB_val']:
                particle['position'] = particle['mixwGB_pos']
            # mixwiPB is best
            elif particle['mixwPB_val'] >= particle['mixwLB_val'] >= particle['mixwGB_val']:
                particle['position'] = particle['mixwPB_pos']
            # particle is best
            else:
                particle['position'] = random_jump(particle['position'])



def optimize(init_pos):

    optimize.counter += 1

    start = time.perf_counter()

    iterations = 500

    gbest_val_list  = []
    gbest_pos_list  = []

    swarm = SIBV3()
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
        if i == iterations-1: # Get the value from the last iteration
            gbest_pos_list.append(swarm.gbest_pos)
            if feasible_vec(swarm.gbest_pos):
                print("Constraints met!")
    
    end = time.perf_counter()
    total_time = end-start

    return gbest_val_list, total_time


if __name__ == '__main__':
    
    # gbest_vals, time_taken = optimize(split_particles_list[0])
    # print(time_taken)
    optimize.counter = 0
    experiment(optimise_func=optimize, split_particles_list=split_particles_list, experiment_name='sibv3_ex_40')

    # gbest_vals, total_time = optimize()
    # plot_results(gbest_vals, total_time)
