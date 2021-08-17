import numpy as np
from dataclasses import dataclass
import time
from pso import PSO, calculate_profit, poss_val, feasible_vec, random_val, demand, plot_results, experiment, rs, split_particles_list
from numba import njit 

@njit
def sib_mix(vec: np.ndarray, better_vec: np.ndarray)-> np.ndarray:
    ''' Takes 2 vectors and returns the mix of the two
            qb: a number of values that are to be replaced - from
            the total differences between the 2 vectors
            diff_index: the qb number ([:qb]) of index values of the absolute 
            differences between the two vectors sorted in descending order([::-1])
        '''
    mix_vec = vec.copy()
    percent_to_replace = 0.10
    qb = int(np.ceil(np.where(mix_vec!=better_vec,1,0).sum()*percent_to_replace))
    diff_index = np.argsort(np.absolute(better_vec - mix_vec))[::-1]
    for _ in range(qb):
        for ind in diff_index:
            if poss_val(index=ind, val=better_vec[ind], vec=mix_vec):
                mix_vec[ind]=better_vec[ind]
            else:
                break
    
    assert feasible_vec(mix_vec), 'sib_mix() returned an unfeasible vector' 
    return mix_vec
    

def random_jump(vec: np.ndarray) -> np.ndarray:
    ''' Replaces a percentage of values from a vector with random values that
        meet demand and supply constraints
        non_zero_inds: index positions that have demand '''    
    global demand
    new_vec = vec.copy()
    percent_to_replace = 0.40
    non_zero_inds = np.where(demand!=0)[0] # yeilds a tuple -> get the first value [0]
    nos_to_replace = int(percent_to_replace * non_zero_inds.size)
    inds = np.random.choice(non_zero_inds, nos_to_replace, replace=False)
    for i in inds:
        new_vec[i] = random_val(vec=new_vec, index=i)
    
    assert feasible_vec(new_vec), 'random_jump() returned an unfeasible vector'
    return new_vec



@dataclass
class SIB(PSO):

    def mix(self):
        for particle in self.particles:
            particle['mixwGB_pos'] = sib_mix(particle['position'], self.gbest_pos)
            particle['mixwGB_val'] = calculate_profit(particle['mixwGB_pos'], sup_cha=rs)
            particle['mixwLB_pos'] = sib_mix(particle['position'], particle['lbest_pos'])
            particle['mixwLB_val'] = calculate_profit(particle['mixwLB_pos'], sup_cha=rs)
    

    def move(self):
        for particle in self.particles:
            if particle['mixwLB_val'] >= particle['mixwGB_val'] >= particle['pbest_val']:
                particle['position'] = particle['mixwLB_pos']
                particle['pbest_val'] = particle['mixwLB_val']
            elif particle['mixwGB_val'] >= particle['mixwLB_val']>= particle['pbest_val']:
                particle['position'] = particle['mixwGB_pos']
                particle['pbest_val'] = particle['mixwGB_val']
            else:
                particle['position'] = random_jump(particle['position'])


def optimize(init_pos):

    start = time.perf_counter()

    iterations = 500

    gbest_val_list  = []
    gbest_pos_list  = []

    swarm = SIB()
    swarm.initialise_with_particle_list(init_pos)
    swarm.pick_informants_ring_topology()
    
    
    for i in range(iterations):
        swarm.calculate_fitness()
        # print([particle['profit']for particle in swarm.particles]) 
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
    
    # print([max([calculate_profit(vec=parts, sup_cha=rs) for parts in spl]) for spl in split_particles_list]) 

    
    experiment(optimize, split_particles_list, "sib_reduced_supply2")

    # gbest_vals, total_time = optimize()
    # plot_results(gbest_vals, total_time)
