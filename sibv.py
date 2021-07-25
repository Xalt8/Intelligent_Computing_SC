import numpy as np
from dataclasses import dataclass
import time
from pso import random_val, poss_val, G, feasible_vec, calculate_profit, plot_results
from sib import SIB


def sibv_mix(position: np.ndarray, better_position: np.ndarray) -> np.ndarray:
    ''' Returns a vector array that is closer to an array with better fitness. 
        Parameters: array(position) and an array with better fitness (better_position)
        If the values of position and better_position are the same then it keeps that value, 
        else a randomly chosen value between the position and better_position is selected.'''
    # new_pos = np.where(better_position==position, better_position, 0)
    new_pos = position.copy()
    percent_to_replace = 0.10
    qb = int(np.ceil(np.where(position!=better_position,1,0).sum()*percent_to_replace))
    diff_index = np.argsort(abs(better_position-position))[::-1]
    for _ in range(qb):
        for ind in diff_index:
            min_val = min(better_position[ind], position[ind])
            max_val = max(better_position[ind], position[ind])
            try:
                r = np.random.randint(min_val, max_val)
                if poss_val(index=ind, val=r, vec=new_pos):
                    new_pos[ind]=r
            except ValueError:
                r = random_val(vec=new_pos, index=ind, graph=G)
                if poss_val(index=ind, val=r, vec=new_pos):
                    new_pos[ind]=r
            else:
                break
        
    assert feasible_vec(new_pos), 'sibv_mix() returned unfeasible vector'  
    return new_pos
    
@dataclass
class SIBV(SIB):
    def mix(self):
        for particle in self.particles:
            particle['mixwGB_pos'] = sibv_mix(particle['position'], self.gbest_pos)
            particle['mixwGB_val'] = calculate_profit(particle['mixwGB_pos'])
            particle['mixwLB_pos'] = sibv_mix(particle['position'], particle['lbest_pos'])
            particle['mixwLB_val'] = calculate_profit(particle['mixwLB_pos'])
    


def optimize():

    start = time.perf_counter()

    iterations = 500

    gbest_val_list  = []
    gbest_pos_list  = []

    swarm = SIBV()
    swarm.initialise()
    swarm.pick_informants_ring_topology()

    for i in range(iterations):
        swarm.calculate_fitness()
        swarm.set_pbest()
        swarm.set_lbest()
        swarm.set_gbest()  
        swarm.mix()
        swarm.move()

        print(f"Iteration: {i} gbest_val: {round(swarm.gbest_val, 2)}")    

        gbest_val_list.append(round(swarm.gbest_val, 2))
        if i == iterations-1: # Get the value from the last iteration
            gbest_pos_list.append(swarm.gbest_pos)
            if feasible_vec(swarm.gbest_pos):
                print("Constraints met!")
    
    end = time.perf_counter()
    total_time = end-start

    return gbest_val_list, total_time


if __name__ == '__main__':
    
    gbest_vals, total_time = optimize()
    plot_results(gbest_vals, total_time)
