import numpy as np
from dataclasses import dataclass
import time
# from pso import random_val, poss_val, feasible_vec, calculate_profit, plot_results, experiment, split_particles_list, rs
from sib_ex import SIB
from pso_ex import PSO, feasible_vec, poss_val, random_val, plot_results, calculate_profit, non_zero_inds, experiment, split_particles_list
from numba import njit 

# @njit
def sibv_mix(position: np.ndarray, better_position: np.ndarray) -> np.ndarray:
    ''' Returns a vector array that is closer to an array with better fitness. 
        Parameters: array(position) and an array with better fitness (better_position)
        If the values of position and better_position are the same then it keeps that value, 
        else a randomly chosen value between the position and better_position is selected.'''
    
    new_pos = position.copy()
    percent_to_replace = 0.40
    qb = int(np.ceil(np.where(position!=better_position,1,0).sum()*percent_to_replace))
    diff_index = np.argsort(np.absolute(better_position-position))[::-1]
    for _ in range(qb):
        for ind in diff_index:
            min_val = np.minimum(better_position[ind], position[ind])
            max_val = np.maximum(better_position[ind], position[ind])
            try:
                r1 = np.random.randint(min_val, max_val)
                if poss_val(index=ind, val=r1, vec=new_pos):
                    new_pos[ind]=r1
                else:
                    r2 = random_val(vec=new_pos, index=ind)
                    if poss_val(index=ind, val=r2, vec=new_pos):
                        new_pos[ind]=r2
            except ValueError: # In case max == 0 
                r2 = random_val(vec=new_pos, index=ind)
                if poss_val(index=ind, val=r2, vec=new_pos):
                    new_pos[ind]=r2
            else:
                break
        
    assert feasible_vec(new_pos), 'sibv_mix() returned unfeasible vector'  
    return new_pos


# @njit
# def sibv_mix(position: np.ndarray, better_position: np.ndarray) -> np.ndarray:
#     ''' Returns a vector array that is closer to an array with better fitness. 
#         Parameters: array(position) and an array with better fitness (better_position)
#         If the values of position and better_position are the same then it keeps that value, 
#         else a randomly chosen value between the position and better_position is selected.'''
    
#     new_pos = position.copy()
#     percent_to_replace = 0.10
#     qb = int(np.ceil(np.where(position!=better_position,1,0).sum()*percent_to_replace))
#     diff_index = np.argsort(np.absolute(better_position-position))[::-1]
#     for _ in range(qb):
#         for ind in diff_index:
#             min_val = np.minimum(better_position[ind], position[ind])
#             max_val = np.maximum(better_position[ind], position[ind])
            
#             if max_val > min_val > 0:
#                 r1 = np.random.randint(min_val, max_val)
#                 if poss_val(index=ind, val=r1, vec=new_pos):
#                     new_pos[ind]=r1
#                 else:
#                     r2 = random_val(vec=new_pos, index=ind)
#                     if poss_val(index=ind, val=r2, vec=new_pos):
#                         new_pos[ind]=r2
#                     else:
#                         break
#             else:
#                 break
        
#     assert feasible_vec(new_pos), 'sibv_mix() returned unfeasible vector'  
#     return new_pos
       
    
@dataclass
class SIBV(SIB):
    def mix(self):
        for particle in self.particles:
            particle['mixwGB_pos'] = sibv_mix(particle['position'], self.gbest_pos)
            particle['mixwGB_val'] = calculate_profit(particle['mixwGB_pos'])
            particle['mixwLB_pos'] = sibv_mix(particle['position'], particle['lbest_pos'])
            particle['mixwLB_val'] = calculate_profit(particle['mixwLB_pos'])


def optimize(init_pos):

    start = time.perf_counter()

    iterations = 500

    gbest_val_list  = []
    gbest_pos_list  = []

    swarm = SIBV()
    swarm.initialise_with_particle_list(init_pos)
    # swarm.initialise()
    swarm.pick_informants_ring_topology()

    for i in range(iterations):
        swarm.calculate_fitness()
        swarm.set_pbest()
        swarm.set_lbest()
        swarm.set_gbest()  
        swarm.mix()
        swarm.move()

        print(f"Iteration: {i} gbest_val: {swarm.gbest_val:.2f}")    

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

    experiment(optimise_func=optimize, split_particles_list=split_particles_list, experiment_name='sibv_ex_40')

    # gbest_vals, total_time = optimize()
    # plot_results(gbest_vals, total_time)
