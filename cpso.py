import numpy as np
from dataclasses import dataclass
import time
from numba import njit
import numba
# from pso import PSO, feasible_vec, poss_val, random_val, plot_results, experiment, split_particles_list
from pso_ex import PSO, feasible_vec, poss_val, random_val, plot_results, experiment, split_particles_list


@njit
def random_back(position: np.ndarray, velocity: np.ndarray)-> np.ndarray:
        ''' Takes a position and a velocity and returns a new position that
            meets demand & supply constraints '''
        vec = position + velocity
        
        if feasible_vec(vec):
            return vec
            
        else:
            new_pos = np.zeros(position.size)
            for i, _ in enumerate(new_pos):
                if poss_val(index = i, val=(int(position[i]+velocity[i])), vec=new_pos):
                    new_pos[i] = int(position[i] + velocity[i])
                else:
                    r = random_val(vec=new_pos, index=i)
                    new_pos[i] = r
            
            assert feasible_vec(vec=new_pos), "random_back() returned an unfeasible vector"  
            return new_pos


@dataclass
class CPSO(PSO):
    
    def set_constricted_velocity(self):
        for particle in self.particles:
            c1 = 2.05
            c2 = 2.05
            ep = c1+c2
            X = 2/(abs(2-ep-np.sqrt((ep**2)-4*ep)))
            dims = particle['position'].shape
            cognitive = (c1 * np.random.uniform(0, 1, dims)*(particle['pbest_pos'] - particle['position']))
            informers = (c2 * np.random.uniform(0, 1, dims)*(particle['lbest_pos'] - particle['position']))
            new_velocity = X * (particle['velocity'] + cognitive + informers)
            particle['velocity'] = new_velocity
    
    
    def move_random_back(self):
        for particle in self.particles:
            new_pos = random_back(particle['position'], particle['velocity'])
            particle['position'] = np.floor(new_pos)


def optimize(init_pos):

    optimize.counter += 1

    start = time.perf_counter()

    iterations = 500

    gbest_val_list  = []
    gbest_pos_list  = []

    swarm = CPSO()
    swarm.initialise_with_particle_list(init_pos)
    # swarm.initialise()
    swarm.pick_informants_ring_topology()

    for i in range(iterations):
        swarm.calculate_fitness()
        swarm.set_pbest()
        swarm.set_lbest()
        swarm.set_gbest()  
        swarm.set_constricted_velocity()
        swarm.move_random_back()

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
    
    optimize.counter = 0
    
    experiment(optimise_func=optimize, split_particles_list=split_particles_list, experiment_name='cpso_ex_19Nov21')

    # gbest_vals, total_time = optimize()
    # plot_results(gbest_vals, total_time)
