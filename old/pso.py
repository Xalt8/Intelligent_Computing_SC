import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import joblib
from dataclasses import dataclass, field
from typing import List
from numba import jit, njit
import cProfile, pstats, io
from graph import SupplyChain


# import data
xls = pd.ExcelFile('data.xlsx')
demand_dict = pd.read_excel(xls, sheet_name='demand', index_col=0, usecols="A:D", nrows=33).T.to_dict('index')  
prices_dict = pd.read_excel(xls, sheet_name = 'prices', index_col=0, usecols="A:D", nrows=33).T.to_dict('index') 
custs = pd.read_excel(xls, sheet_name='customers', index_col=0, usecols="A:C", nrows=33).to_dict('index')
farms_dict = pd.read_excel(xls, sheet_name='farms', index_col=0, usecols="A:E", nrows=31).to_dict('index')
products_dict = {'P1': 6, 'P2': 10, 'P3': 12}
transport_cost_per_egg = {'North':0.10, 'South':0.15}




# Supply chain with equilibrium demand
# sc = SupplyChain(farms=farms_dict, customers=custs, products=products_dict, 
#                 demand=demand_dict, prices=prices_dict, transport=transport_cost_per_egg)

demand_df = pd.read_excel(xls, sheet_name='demand', index_col=0, usecols="A:D", nrows=33).T

## Reduced demand 
reduced_demand = demand_df.apply(lambda x:x*.90).astype(np.int64).to_dict('index')
# Supply chain with reduced demand
# rd = SupplyChain(farms=farms_dict, customers=custs, products=products_dict, 
#                 demand=reduced_demand, prices=prices_dict, transport=transport_cost_per_egg)


reduced_supply = demand_df.apply(lambda x:x*1.10).astype(np.int64).to_dict('index')

rs = SupplyChain(farms=farms_dict, customers=custs, products=products_dict, 
                demand=reduced_supply, prices=prices_dict, transport=transport_cost_per_egg)




## GLOBAL VARIABLES
demand, supply, dims, prod_cap = rs.get_global_variables()



# Functions
@jit(nopython=True)
def feasible_vec(vec:np.ndarray) -> bool:
    '''Returns true if a vec meets demand & supply constraints'''
    global demand, supply, prod_cap, dims
    # prod_cap = np.array([G.nodes[p]['eggs_per_box'] for p in prods])
    mat = vec.reshape(dims)
    supply_check = np.sum(mat * np.expand_dims(prod_cap,axis=1)) <= supply # Check for eggs
    demand_check = np.all((vec <= demand) & (vec >= 0)) # Check boxes
    return demand_check and supply_check


@jit(nopython=True)
def poss_val(index:int, val:int, vec: np.ndarray):
    ''' Returns True if the 'val' being placed in 
        'index' position of 'vec' meets 'demand' and 'supply' 
        constraints '''
    vec_copy = vec.copy()
    vec_copy[index]=val
    return feasible_vec(vec_copy)


@jit(nopython=True)
def random_val(vec:np.ndarray, index: int) -> np.int64:
    ''' Returns a random value that meets demand and supply constraints 
        vec: a vector in which the random value is to placed
        index: the index position in the vector for which the random value is needed
        graph: the supply chain graph '''
    global demand, supply, prod_cap, dims
    
    if demand[index]==0:
        return 0
    else:
        mat = vec.reshape(dims)
                
        # In-place of unravel index - gets the row, col index if reshaped to matrix
        mat_index = np.arange(0, vec.size).reshape(dims)
        row, col = np.where(mat_index == index)
        
        alloc_supply = np.sum(mat * np.expand_dims(prod_cap,axis=1))
        available_supply_eggs = supply - (alloc_supply - (vec[index]* prod_cap[row]))
        available_supply_boxes = np.floor((available_supply_eggs / prod_cap[row])).astype(np.int64)
        if  available_supply_boxes and demand[index] > 0:
            return np.random.randint(0, np.minimum(available_supply_boxes.item(), demand[index].item()))
        else:
            return 0


def random_initiate_vec() -> np.array:
    ''' Returns a vector in the same size of demand that meets demand 
        & supply constraints '''
    global demand
    zero_vec = np.zeros(demand.size)
    indices = np.arange(0,demand.size)
    random.shuffle(indices)
    for i in indices:
        r = random_val(zero_vec, i)
        zero_vec[i]=r
    
    assert feasible_vec(zero_vec), 'random_initiate_vec() returned an unfeasible vector'
    return zero_vec


def calculate_profit(vec: np.ndarray, sup_cha:SupplyChain) -> float:
    ''' Returns the total profit from a given quantities of products '''
    
    cost_of_eggs = sup_cha.get_supply_costs(vec=vec) # -> float
    prices = sup_cha.get_price_per_product() # -> np.ndarray 
    sales = np.sum(prices * vec) # -> float
    transport_cost = sup_cha.get_transport_cost(vec) # -> float
    total_cost = transport_cost + cost_of_eggs
    profit = np.round((sales - total_cost), decimals=3)
    return profit  


def plot_results(gbest_vals, total_time):
    _, ax = plt.subplots() 
    x_axis_vals = [x for x in range(len(gbest_vals))]
    ax.plot(x_axis_vals, gbest_vals)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Profit")
    props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    props2 = dict(boxstyle='round', facecolor='thistle', alpha=0.5)
    ax.text(0.1, 0.9, 'last gbest_val: '+str(round(gbest_vals[-1],2)), transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props1)
    ax.text(0.1, 0.7, 'total_time: '+str(round(total_time,2))+' seconds' , transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props2)
    # ax.set_xlim([0,gbest_vals.size])
    # ax.set_ylim([min(gbest_vals)-10, 10+max(gbest_vals)])
    ax. ticklabel_format(useOffset=False, style='plain')
    plt.tight_layout()
    plt.show()


def profile(fnc):
    """ A decorator that uses cProfile to profile a function
        Source: Sebastiaan Mathôt https://osf.io/upav8/
    """
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def already_there(list_of_particles: list, new_particle: np.ndarray):
    ''' Checks to see if a new_particle is present in list_of_particles'''
    return np.any([np.array_equal(new_particle, particle) for particle in list_of_particles])


def generate_particle_list(num_of_particles:int)-> list:
    ''' Generates a list of unique particles'''
    particle_list = []
    while len(particle_list) < num_of_particles:
        vec = random_initiate_vec()
        if not already_there(particle_list, vec):
            particle_list.append(vec)
        else:
            print("Already there!")
    return particle_list


def split_list(particle_list:list, num_particles:int) -> list:
    ''' Takes a list of particles and splits it by 
        the num_particles returns a list of lists'''
    return [particle_list[i:i+num_particles] for i in range(0, len(particle_list), num_particles)]


def make_result_matrix(data) -> pd.DataFrame:
    ''' Takes the results data and converts it into a 
        DataFrame'''
    # 1 for time float + len of data for gbest_vals
    size = (1 + len(data[0][0]), len(data))
    arr = np.zeros(size)
    for i, d in enumerate(data):
        arr[:len(d[0]),i] = d[0] # put gbest_vals 
        arr[-1,:] = d[1] # put time data in the last row
    df = pd.DataFrame(arr, columns =[str(i)+"_run" for i in range(len(data))])
    as_list = df.index.tolist()
    as_list[-1] = "Time" # Make the index value of last row to time
    df.index = as_list
    return df


def experiment(optimise_func, split_particles_list:list, experiment_name:str):
    ''' Applies an optimisation function to all particles (init_pos) in split_particle_list  
        Creates a dataframe of the results and saves it to an excel file. 
    '''
    results = [optimise_func(init_pos) for init_pos in split_particles_list]

    experiment_results = make_result_matrix(results)
    experiment_results.to_excel(f"experiment_result_{experiment_name}.xlsx")
    return experiment_results



@dataclass
class PSO:
    num_particles: int = 20
    particles: List = field(init=False)
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
    
    
    # def set_gbest(self):
    #     for particle in self.particles:
    #         if particle['lbest_val'] >= self.gbest_val:
    #             self.gbest_val = particle['lbest_val']
    #             self.gbest_pos = particle['lbest_pos']
    
    def set_gbest(self):
        for i, particle in enumerate(self.particles):
            if particle['lbest_val'] >= self.gbest_val:
                self.gbest_val = particle['lbest_val']
                self.gbest_pos = particle['lbest_pos']
                self.gbest_particle_index = i # Index number of the gbest particle


    def set_overwrite_particles_list(self):
        ''' Puts the index numbers of the first half of particles in a swarm
            in a list except for the gbest particle index '''
        self.overwrite_particles_list = [
            j for j in range(int(self.num_particles/2)) 
                if j!=self.gbest_particle_index]
    

    def overwrite_particle(self, vec:np.ndarray):
        ''' Replaces a randomly chosen particle's position with a given vector'''
        particle_to_overwrite = random.choice(self.overwrite_particles_list)
        self.particles[particle_to_overwrite]['position'] = vec

    



def get_particle_list(joblib_lists:str, pso_alg:PSO, sup_chn:SupplyChain)-> list:
    particle_list = joblib.load(joblib_lists)
    split_particles_list = split_list(particle_list, pso_alg.num_particles)    
    for list_of_particles in split_particles_list:
        list_of_particles.pop()
        list_of_particles.append(sup_chn.fill_order())
    
    return split_particles_list



split_particles_list = get_particle_list(joblib_lists='particle_list3', pso_alg=PSO, sup_chn=rs)    
    
# particle_list = joblib.load('particle_list3')
# split_particles_list = split_list(particle_list, PSO.num_particles)



if __name__ =='__main__':
    pass