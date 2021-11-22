import numpy as np
from dataclasses import dataclass
import time
from pso_ex import PSO, feasible_vec, calculate_profit, experiment, split_particles_list
from sib3 import random_jump
from sibv3_ex import get_r1r2
from numba import njit 


@njit
def sibv_mix2(vec: np.ndarray, better_vec: np.ndarray, percent_to_replace:int) -> np.ndarray:
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
    # percent_to_replace = 0.20
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
class SIBV4(PSO):
    def mix(self):
        for particle in self.particles:
            particle['mixwLB_pos'] = sibv_mix2(vec=particle['position'], better_vec=particle['lbest_pos'], percent_to_replace=0.40)
            particle['mixwLB_val'] = calculate_profit(particle['mixwLB_pos'])
            particle['mixwGB_pos'] = sibv_mix2(vec=particle['position'], better_vec=self.gbest_pos, percent_to_replace=0.30)
            particle['mixwGB_val'] = calculate_profit(particle['mixwGB_pos'])
            

    def move(self):
        for particle in self.particles:
            # mixwLB is best
            if (particle['mixwLB_val'] > particle['mixwGB_val']) and (particle['mixwLB_val'] > particle['pbest_val']):
                particle['position'] = particle['mixwLB_pos']
            # mixwGB is best
            elif (particle['mixwGB_val'] > particle['mixwLB_val']) and (particle['mixwGB_val'] > particle['pbest_val']):
                particle['position'] = particle['mixwGB_pos']
            # particle is best
            else:
                particle['position'] = random_jump(particle['position'])



def optimize(init_pos):

    optimize.counter += 1

    start = time.perf_counter()

    iterations = 500

    gbest_val_list  = []
    gbest_pos_list  = []

    swarm = SIBV4()
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
    
    optimize.counter = 0
    experiment(optimise_func=optimize, split_particles_list=split_particles_list, experiment_name='sibv4_40_30')