import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass, field
from numba import njit
from graph_ex import sc, sum_to_num


# Global variables
demand_indices, split_list = sc.get_demand_check_variables()
sup_indices, sup_split_list, sup_packs = sc.get_supply_check_variables()
eggs_supply = sc.get_eggs_supplied()
demand = sc.get_demand_vec()


# Functions
@njit
def feasible_vec(vec:np.ndarray)-> bool:
    ''' Checks to see if the vec meets demand and supply constraints '''
    global sup_indices, sup_split_list, sup_packs, eggs_supply, demand, demand_indices, split_list

    # Supply check    
    vec_sum = np.array([np.sum(vec[sup_indices[i]] * sup_packs[i]) 
                for i in np.arange(len(sup_indices))], dtype=np.int64)
    vec_sum_split = np.split(vec_sum, sup_split_list)[:-1]
    farm_wise_totals = np.array([np.sum(vec_sum_split[i]) for i in range(len(vec_sum_split))], dtype=np.int64)
    supply_check = np.all(farm_wise_totals <= eggs_supply)    

    # Demand check
    split_array = np.split(demand_indices, split_list)[:-1]
    vec_sum = np.array([np.sum(vec[i]) for i in split_array])
    demand_check = np.all(vec_sum <= demand)   
    
    # Zero check
    zero_check = np.all(vec >= 0)

    return demand_check and zero_check and supply_check
    

@njit
def poss_val(index:int, val:int, vec: np.ndarray):
    ''' Returns True if the 'val' being placed in 
        'index' position of 'vec' meets 'demand' and 'supply' 
        constraints '''
    vec_copy = vec.copy()
    vec_copy[index]=val
    return feasible_vec(vec_copy)


@njit
def get_available_demand(vec:np.ndarray, index:int) -> int:
    ''' Returns the available demand for an index in an vector'''
    
    global demand_indices, split_list

    split_array = np.split(demand_indices, split_list)[:-1]
    for cprod_indices, dem in zip(split_array, demand):
        if index in cprod_indices:
            vec_cprod = np.array([vec[i] for i in cprod_indices], dtype=np.int64)
            
            available_demand = dem - (np.sum(vec_cprod) - vec_cprod[np.where(cprod_indices==index)])
            avail_demand = np.maximum(0, available_demand)[0] # Don't return negative demand
    return avail_demand


@njit
def get_availble_supply(vec:np.ndarray, index:int) -> int:
    global sup_indices, sup_split_list, sup_packs, eggs_supply
    
    sup_inices_split = np.split(sup_indices, sup_split_list)
    sup_packs_split = np.split(sup_packs, sup_split_list)

    loc = np.array([i for i in range(len(sup_inices_split)) if index in sup_inices_split[i]], dtype=np.int64)[0]  
    row, _ = np.where(sup_inices_split[loc] == index)

    arr_copy = np.where(sup_inices_split[loc] == index, 0, sup_inices_split[loc])
    supplied = np.sum(np.array([np.sum(vec[arr_copy[arr_row_ind]] * sup_packs_split[loc][arr_row_ind]) 
                for arr_row_ind in np.arange(arr_copy.shape[0])], dtype=np.int64))

    available_eggs = eggs_supply[loc] - supplied
    avail_supply = (available_eggs/sup_packs_split[loc][row]).astype(np.int64)


    return np.maximum(0, avail_supply)


@njit
def random_val(vec:np.ndarray, index: int) -> np.int64:
    available_supply = get_availble_supply(vec=vec, index=index)
    available_demand = get_available_demand(vec=vec, index=index)
    
    return np.random.randint(0, np.minimum(available_demand, available_supply))


def random_instantiate_vec():
    global sc
    zero_vec = np.zeros(len(sc.get_optimising_tuples()))
    fprods = [fprod for farm in sc.farms.keys() for fprod in sc.graph.successors(farm)]
    fprod_pack_dict = {fprod:pack for fprod, pack in zip(fprods, sc.pack_random_eggs())}

    for fprod, pack in fprod_pack_dict.items():
        cprods = [cprod for cprod in sc.graph.successors(fprod)]
        dist = sum_to_num(pack, len(cprods))
        for cprod, qty in zip(cprods, dist):
            index = sc.get_indices_dict().get((fprod, cprod), -1)
            if poss_val(index=index, val=qty, vec=zero_vec):
                zero_vec[index]=qty
            else:
                r = random_val(vec= zero_vec, index = index)
                zero_vec[index]= r
    
    assert feasible_vec(vec=zero_vec), 'random_instantiate_vec() returned an unfeasible vec '
    return zero_vec



def plot_results(gbest_vals, total_time):
    _, ax = plt.subplots() 
    x_axis_vals = [x for x in range(len(gbest_vals))]
    ax.plot(x_axis_vals, gbest_vals)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Profit")
    props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    props2 = dict(boxstyle='round', facecolor='thistle', alpha=0.5)
    ax.text(0.1, 0.9, 'last gbest_val: '+str(round(gbest_vals[-1],2)), 
            transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props1)
    ax.text(0.1, 0.7, 'total_time: '+str(round(total_time,2))+' seconds', 
            transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props2)
    # ax.set_xlim([0,gbest_vals.size])
    # ax.set_ylim([min(gbest_vals)-10, 10+max(gbest_vals)])
    ax. ticklabel_format(useOffset=False, style='plain')
    plt.tight_layout()
    plt.show()




@dataclass
class PSO:
    num_particles: int = 20
    particles: list = field(init=False)
    gbest_val: float = -np.Inf
    gbest_pos: np.array = field(init=False)


    def __post_init__(self):
        self.particles = [dict() for _ in range(self.num_particles)]


    def initialise(self):
        for particle in self.particles:
            particle['position'] = random_initiate_vec()
            particle['pbest_val'] = -np.Inf
            particle['velocity'] = np.zeros(particle['position'].size)
    

    def initialise_with_particle_list(self, particle_pos_list):
        for particle, pos in zip(self.particles, particle_pos_list):
            particle['position'] = pos
            particle['pbest_val'] = -np.Inf
            particle['velocity'] = np.zeros(particle['position'].size)
    

    def pick_informants_ring_topology(self):
        for index, particle in enumerate(self.particles):
            particle['informants'] = []
            particle['lbest_val'] = -np.Inf
            particle['informants'].append(self.particles[(index-1) % len(self.particles)])
            particle['informants'].append(self.particles[index])
            particle['informants'].append(self.particles[(index+1) % len(self.particles)])
    

    def calculate_fitness(self):
        for particle in self.particles:
            particle['profit'] = calculate_profit(particle['position'], sup_cha=rs)

    
    def set_pbest(self):
        for particle in self.particles:
            if particle['profit'] > particle['pbest_val']:
                particle['pbest_val'] = particle['profit']
                particle['pbest_pos'] = particle['position']
    

    def set_lbest(self):
        for particle in self.particles:
            for informant in particle['informants']:
                if(informant['pbest_val'] >= particle['lbest_val']):
                    informant['lbest_val'] = particle['pbest_val']
                    informant['lbest_pos'] = particle['pbest_pos']
    
    
    def set_gbest(self):
        for particle in self.particles:
            if particle['lbest_val'] >= self.gbest_val:
                self.gbest_val = particle['lbest_val']
                self.gbest_pos = particle['lbest_pos']
    
    


if __name__ =='__main__':
    random_instantiate_vec()