import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import joblib
from dataclasses import dataclass, field
from typing import List

# Import data
farms = joblib.load('farms_list')
custs = joblib.load('customer_list')
prods = joblib.load('product_list')
transport_cost_per_egg = joblib.load('transport_costs')
G = joblib.load('supply_chain_graph')


# Getters
def get_total_supply(graph: nx.DiGraph) -> int:
    ''' Return the total number of eggs supplied by farms to dealer'''
    return np.sum([graph[farm]['Dealer']['quantity'] for farm in graph.predecessors('Dealer')])

def get_demand(graph: nx.DiGraph) -> np.ndarray:
    ''' Gets a demand array from graph'''
    global prods
    return np.array([graph[p][cust]['demand'] for p in prods for cust in graph.successors(p)]).astype(np.int64)   


def get_transport_costs(vec: np.array, graph: nx.DiGraph) -> float:
    ''' Returns the total transport cost for a given array of quantities '''
    global prods, custs, transport_cost_per_egg
    prod_cap = np.array([graph.nodes[p]['eggs_per_box'] for p in prods])
    per_egg_costs =  np.array([transport_cost_per_egg[graph.nodes[c]['location']] for p in prods for c in G.successors(p)])
    per_egg_mat = per_egg_costs.reshape(len(prods), len(custs))
    mat = vec.reshape(len(prods), len(custs))
    return np.sum(mat * per_egg_mat * prod_cap[:, None])


def get_supply_costs(graph: nx.DiGraph, vec: np.ndarray)-> float:
    ''' Gets the total cost of eggs supplied given a vector of product quantities'''
    global prods, custs
    avg_cost_per_egg = np.mean([graph[farm]['Dealer']['cost_per_egg'] for farm in graph.predecessors('Dealer')])
    mat = vec.reshape(len(prods), len(custs))
    prod_cap = np.array([graph.nodes[p]['eggs_per_box'] for p in prods])
    total_eggs = np.sum(mat * prod_cap[:, None])
    return total_eggs * avg_cost_per_egg


def get_price_per_product(graph: nx.DiGraph) -> np.ndarray:
    ''' Returns the prices for the products '''
    return np.array([graph[p][c]['price'] for p in prods for c in graph.successors(p)])


supply = get_total_supply(G)
demand = get_demand(G)


# Functions
def feasible_vec(vec:np.ndarray) -> bool:
    '''Returns true if a vec meets demand & supply constraints'''
    global demand, supply, prods, G
    prod_cap = np.array([G.nodes[p]['eggs_per_box'] for p in prods])
    mat = vec.reshape(len(prods), len(custs))
    supply_check = np.sum(mat * prod_cap[:, None]) <= supply # Check for eggs
    demand_check = np.all((vec <= demand) & (vec >= 0)) # Check boxes
    return demand_check and supply_check


def poss_val(index:int, val:int, vec: np.ndarray):
    ''' Returns True if the 'val' being placed in 
        'index' position of 'vec' meets 'demand' and 'supply' 
        constraints '''
    vec_copy = vec.copy()
    vec_copy[index]=val
    return feasible_vec(vec_copy)


def get_supply_boxes(vec: np.ndarray) -> np.array:
    ''' Returns an array with boxes of products '''
    global prods, custs
    return np.sum(vec.reshape(len(prods), len(custs)), axis=1)


def random_val(vec:np.ndarray, index: int, graph: nx.DiGraph) -> int:
    ''' Returns a random value that meets demand and supply constraints 
        vec: a vector in which the random value is to placed
        index: the index position in the vector for which the random value is needed
        graph: the supply chain graph '''
    global demand, supply, prods, custs
    
    if demand[index]==0:
        return 0
    else:
        mat = vec.reshape(len(prods), len(custs))
        prod_cap = np.array([graph.nodes[p]['eggs_per_box'] for p in prods])
        
        # In-place of unravel index - gets the row, col index if reshaped to matrix
        mat_index = np.arange(0, vec.size).reshape(len(prods), len(custs))
        row, col = np.where(mat_index == index)
        
        alloc_supply = np.sum(mat * prod_cap[:, None])
        available_supply_eggs = supply - (alloc_supply - (vec[index]* prod_cap[row]))
        available_supply_boxes = int(available_supply_eggs // prod_cap[row])
        if  available_supply_boxes and demand[index] > 0:
            return np.random.randint(min(available_supply_boxes, demand[index]))
        else:
            return 0


def random_initiate_vec() -> np.array:
    ''' Returns a vector in the same size of demand that meets demand 
        & supply constraints '''
    global demand, G
    zero_vec = np.zeros(demand.size)
    indices = np.arange(0,demand.size)
    random.shuffle(indices)
    for i in indices:
        r = random_val(zero_vec, i, G)
        zero_vec[i]=r
    return zero_vec


def calculate_profit(vec: np.ndarray) -> float:
    ''' Returns the total profit from a given quantities of products'''
    global G
    cost_of_eggs = get_supply_costs(graph = G, vec=vec) # -> float
    prices = get_price_per_product(G) # -> np.ndarray 
    sales = np.sum(prices * vec) # -> float
    transport_cost = get_transport_costs(vec, G) # -> float
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