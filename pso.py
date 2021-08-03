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



# # Import data
# farms = joblib.load('farms_list')
# custs = joblib.load('customer_list')
# prods = joblib.load('product_list')
# transport_cost_per_egg = joblib.load('transport_costs')
# G = joblib.load('supply_chain_graph')


# # Getters
# def get_total_supply(graph: nx.DiGraph) -> int:
#     ''' Return the total number of eggs supplied by farms to dealer'''
#     return np.sum([graph[farm]['Dealer']['quantity'] for farm in graph.predecessors('Dealer')])


# def get_demand(graph: nx.DiGraph) -> np.ndarray:
#     ''' Gets a demand array from graph'''
#     global prods
#     return np.array([graph[p][cust]['demand'] for p in prods for cust in graph.successors(p)]).astype(np.int64)   


# def get_transport_costs(vec: np.array, graph: nx.DiGraph) -> float:
#     ''' Returns the total transport cost for a given array of quantities '''
#     global prods, custs, transport_cost_per_egg
#     prod_cap = np.array([graph.nodes[p]['eggs_per_box'] for p in prods])
#     per_egg_costs =  np.array([transport_cost_per_egg[graph.nodes[c]['location']] for p in prods for c in G.successors(p)])
#     per_egg_mat = per_egg_costs.reshape(len(prods), len(custs))
#     mat = vec.reshape(len(prods), len(custs))
#     return np.sum(mat * per_egg_mat * np.expand_dims(prod_cap,axis=1))


# def get_supply_costs(graph: nx.DiGraph, vec: np.ndarray)-> float:
#     ''' Gets the total cost of eggs supplied given a vector of product quantities'''
#     global prods, custs
#     avg_cost_per_egg = np.mean([graph[farm]['Dealer']['cost_per_egg'] for farm in graph.predecessors('Dealer')])
#     mat = vec.reshape(len(prods), len(custs))
#     prod_cap = np.array([graph.nodes[p]['eggs_per_box'] for p in prods])
#     total_eggs = np.sum(mat * np.expand_dims(prod_cap,axis=1))
#     return total_eggs * avg_cost_per_egg


# def get_price_per_product(graph: nx.DiGraph) -> np.ndarray:
#     ''' Returns the prices for the products '''
#     return np.array([graph[p][c]['price'] for p in prods for c in graph.successors(p)])


# def get_supply_boxes(vec: np.ndarray) -> np.array:
#     ''' Returns an array with boxes of products '''
#     global prods, custs
#     return np.sum(vec.reshape(len(prods), len(custs)), axis=1)


# def get_supplied_eggs(vec: np.ndarray)-> int:
#     ''' Returns the number of eggs supplied to customers based on 
#         quantity of boxes (vec) '''
#     global prod_cap
#     boxes = get_supply_boxes(vec)
#     return np.sum([np.sum(box * pc) for box, pc in zip(boxes, prod_cap)])


# ### Derived global variables
# supply = get_total_supply(G)
# demand = get_demand(G)
# prod_cap = np.array([G.nodes[p]['eggs_per_box'] for p in prods])
# mat_shape = (len(prods), len(custs))

# import data
xls = pd.ExcelFile('data.xlsx')
demand_dict = pd.read_excel(xls, sheet_name='demand', index_col=0, usecols="A:D", nrows=33).T.to_dict('index')  
prices_dict = pd.read_excel(xls, sheet_name = 'prices', index_col=0, usecols="A:D", nrows=33).T.to_dict('index') 
custs = pd.read_excel(xls, sheet_name='customers', index_col=0, usecols="A:C", nrows=33).to_dict('index')
farms_dict = pd.read_excel(xls, sheet_name='farms', index_col=0, usecols="A:E", nrows=31).to_dict('index')
products_dict = {'P1': 6, 'P2': 10, 'P3': 12}
transport_cost_per_egg = {'North':0.10, 'South':0.15}


sc = SupplyChain(farms=farms_dict, customers=custs, products=products_dict, 
                demand=demand_dict, prices=prices_dict, transport=transport_cost_per_egg)


demand = sc.get_demand_vec()
supply = sc.get_total_supply()
dims = (len(sc.products), len(sc.customers))
prod_cap = np.array([sc.graph.nodes[p]['eggs_per_box'] for p in sc.products])


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
        #ValueError: total size of new array must be unchanged
        mat = vec.reshape(dims)
                
        # In-place of unravel index - gets the row, col index if reshaped to matrix
        mat_index = np.arange(0, vec.size).reshape(dims)
        row, col = np.where(mat_index == index)
        
        alloc_supply = np.sum(mat * np.expand_dims(prod_cap,axis=1))
        available_supply_eggs = supply - (alloc_supply - (vec[index]* prod_cap[row]))
        available_supply_boxes = np.floor((available_supply_eggs / prod_cap[row])).astype(np.int64)
        if  available_supply_boxes and demand[index] > 0:
            return np.random.randint(np.minimum(available_supply_boxes.item(), demand[index].item()))
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
    return zero_vec


def calculate_profit(vec: np.ndarray) -> float:
    ''' Returns the total profit from a given quantities of products '''
    global sc
    cost_of_eggs = sc.get_supply_costs(vec=vec) # -> float
    prices = sc.get_price_per_product() # -> np.ndarray 
    sales = np.sum(prices * vec) # -> float
    transport_cost = sc.get_transport_cost(vec) # -> float
    total_cost = transport_cost + cost_of_eggs
    profit = sales - total_cost
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
    
    
    def pick_informants_ring_topology(self):
        for index, particle in enumerate(self.particles):
            particle['informants'] = []
            particle['lbest_val'] = -np.Inf
            particle['informants'].append(self.particles[(index-1) % len(self.particles)])
            particle['informants'].append(self.particles[index])
            particle['informants'].append(self.particles[(index+1) % len(self.particles)])
    

    def calculate_fitness(self):
        for particle in self.particles:
            particle['profit'] = calculate_profit(particle['position'])

    
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
    pass