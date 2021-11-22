from dataclasses import dataclass, field
from graph import SupplyChain
import numpy as np
from pso import PSO, split_particles_list, plot_results, rs, feasible_vec, calculate_profit, profile, plot_results, split_particles_list
from cpso import CPSO
from sib import SIB
from sibv import SIBV
import time, random
# from numba import njit



np.set_printoptions(suppress=True)


class PSO2(SIB, CPSO, PSO):

    def __init__(self, supply_chain:SupplyChain, product:str):
        self.supply_chain = supply_chain
        self.product = product
        self.demand = self.supply_chain.get_demand_by_product(self.product)
        self.eggs_per_box = supply_chain.products[product]
        self.supply = self.supply_chain.allocate_supply_by_product()[self.product]
        
        super().__init__()
    
    def __repr__(self):
        return f'{self.__class__.__name__}(Prod:{self.product}, gbest_val:{self.gbest_val})'


    def feasible_vec(self, vec:np.ndarray) -> bool:
        '''Returns true if a vec meets demand & supply constraints'''
        
        supply_check = np.sum(vec * self.eggs_per_box) <= self.supply # Check for eggs
        demand_check = np.all((vec <= self.demand) & (vec >= 0)) # Check boxes
        
        return demand_check and supply_check

    
    def poss_val(self, index:int, val:int, vec: np.ndarray):
        ''' Returns True if the 'val' being placed in 
            'index' position of 'vec' meets 'demand' and 'supply' 
            constraints '''
        vec_copy = vec.copy()
        vec_copy[index]=val
        return self.feasible_vec(vec_copy)

    
    def random_val(self, vec:np.ndarray, index: int) -> np.int64:
        ''' Returns a random value that meets demand and supply constraints 
            vec: a vector in which the random value is to placed
            index: the index position in the vector for which the random value is needed
            graph: the supply chain graph '''
        # eggs_per_box = supply_chain.products[prod_name]
        
        if self.demand[index]==0:
            return 0
        else:
            alloc_supply = np.sum(vec * self.eggs_per_box)
            available_supply_eggs = self.supply - (alloc_supply - (vec[index]* self.eggs_per_box))
            available_supply_boxes = np.floor((available_supply_eggs / self.eggs_per_box)).astype(np.int64)
            if np.minimum(available_supply_boxes.item(), self.demand[index].item()) > 0:
                return np.random.randint(0, np.minimum(available_supply_boxes.item(), self.demand[index].item()))
            else:
                return 0

    
    def random_initiate_vec(self) -> np.array:
        ''' Returns a vector in the same size of demand that meets demand 
            & supply constraints '''
        
        zero_vec = np.zeros(self.demand.size)
        indices = np.arange(0, self.demand.size)
        random.shuffle(indices)
        for i in indices:
            r = self.random_val(vec=zero_vec, index=i)
            zero_vec[i]=r
        
        assert self.feasible_vec(zero_vec), 'random_initiate_vec() returned an unfeasible vector'
        return zero_vec


    def initialise(self):
        for particle in self.particles:
            particle['position'] = self.random_initiate_vec()
            particle['pbest_val'] = -np.Inf
            particle['velocity'] = np.zeros(particle['position'].size)
    

    def calculate_profit2(self, vec:np.ndarray) -> float:
        cost_of_eggs = self.supply_chain.get_supply_costs_by_product(vec=vec, product=self.product) # -> float
        prices = np.array([self.supply_chain.graph[p][c]['price'] for p in self.supply_chain.products 
                for c in self.supply_chain.graph.successors(p) if p==self.product]) 
        assert prices.size == vec.size, "prices & vec are don't have the same dims!" 
        sales = np.sum(prices * vec) # -> float
        transport_cost = self.supply_chain.get_transport_cost_by_product(vec, self.product) # -> float
        total_cost = transport_cost + cost_of_eggs
        profit = np.round((sales - total_cost), decimals=3)
        return profit  

    
    def calculate_fitness2(self):
        for particle in self.particles:
            particle['profit'] = self.calculate_profit2(particle['position'])

    
    def random_back2(self, position:np.ndarray, velocity:np.ndarray,)-> np.ndarray:
        ''' Takes a position and a velocity and returns a new position that
            meets demand & supply constraints '''
        vec = position + velocity
        
        if self.feasible_vec(vec):
            return vec
            
        else:
            new_pos = np.zeros(position.size)
            for i, _ in enumerate(new_pos):
                if self.poss_val(i, (int(position[i]+velocity[i])), new_pos):
                    new_pos[i] = int(position[i] + velocity[i])
                else:
                    r = self.random_val(vec=new_pos, index=i)
                    new_pos[i] = r
            
            assert self.feasible_vec(vec=new_pos), "random_back2() returned an unfeasible vector"  
            return new_pos
    

    def move_random_back(self):
        for particle in self.particles:
            new_pos = self.random_back2(particle['position'], particle['velocity'])
            particle['position'] = np.floor(new_pos)



    def sib_mix(self, vec:np.ndarray, better_vec:np.ndarray):

        mix_vec = vec.copy()
        percent_to_replace = 0.10
        qb = int(np.ceil(np.where(mix_vec!=better_vec,1,0).sum()*percent_to_replace))
        diff_index = np.argsort(np.absolute(better_vec - mix_vec))[::-1]
        for _ in range(qb):
            for ind in diff_index:
                if self.poss_val(index=ind, val=better_vec[ind], vec=mix_vec):
                    mix_vec[ind]=better_vec[ind]
                else:
                    break
                     
                    
        assert self.feasible_vec(mix_vec), 'sib_mix() returned an unfeasible vector' 
        return mix_vec
            

    def mix(self):
        for particle in self.particles:
            particle['mixwGB_pos'] = self.sib_mix(particle['position'], self.gbest_pos)
            particle['mixwGB_val'] = self.calculate_profit2(particle['mixwGB_pos'])
            particle['mixwLB_pos'] = self.sib_mix(particle['position'], particle['lbest_pos'])
            particle['mixwLB_val'] = self.calculate_profit2(particle['mixwLB_pos'])
    

    def random_jump(self, vec:np.ndarray) -> np.ndarray:
        ''' Replaces a percentage of values from a vector with random values that
            meet demand and supply constraints
            non_zero_inds: index positions that have demand '''    
        
        new_vec = vec.copy()
        percent_to_replace = 0.40
        non_zero_inds = np.where(self.demand!=0)[0] # yeilds a tuple -> get the first value [0]
        nos_to_replace = int(percent_to_replace * non_zero_inds.size)
        inds = np.random.choice(non_zero_inds, nos_to_replace, replace=False)
        for i in inds:
            new_vec[i] = self.random_val(vec=new_vec, index=i)
        
        assert self.feasible_vec(new_vec), 'random_jump() returned an unfeasible vector'
        return new_vec
    

    def move(self):
        for particle in self.particles:
            if particle['mixwLB_val'] >= particle['mixwGB_val'] >= particle['pbest_val']:
                particle['position'] = particle['mixwLB_pos']
                particle['pbest_val'] = particle['mixwLB_val']
            elif particle['mixwGB_val'] >= particle['mixwLB_val']>= particle['pbest_val']:
                particle['position'] = particle['mixwGB_pos']
                particle['pbest_val'] = particle['mixwGB_val']
            else:
                particle['position'] = self.random_jump(particle['position'])
    

    def split_gbest_vec(self, vec:np.ndarray) -> list:
        return np.split(vec, len(self.supply_chain.products))


def main():

    start = time.perf_counter()
    gbest_val_list  = []

    iterations = 500

    # Instantiate the dimensional PSO       
    psos2 = [PSO2(supply_chain=rs, product=prod) for prod in rs.products.keys()]
    # Instantiate the regular PSO
    # pso1 = CPSO()
    pso1 = SIB()
    # pso1 = SIBV()
    
    pso1.initialise_with_particle_list(split_particles_list[0])
    pso1.pick_informants_ring_topology()

    for pso2 in psos2:
        pso2.initialise()
        pso2.pick_informants_ring_topology()
    
    
    # Main loop
    for i in range(iterations):
        
        pso1.calculate_fitness()
        pso1.set_pbest()
        pso1.set_lbest()
        pso1.set_gbest()
        pso1.set_overwrite_particles_list()
        

        for pso2 in psos2:
            pso2.calculate_fitness2()
            pso2.set_pbest()
            pso2.set_lbest()
            pso2.set_gbest()
            pso2.set_overwrite_particles_list()

        context_vec = np.concatenate([pso2.gbest_pos for pso2 in psos2])
        assert feasible_vec(context_vec), 'Context vec unfeasible!'

        # Overwrite a particle in the regular PSO
        pso1.overwrite_particle(vec=context_vec)

        # Take the gbest_pos from the regular PSO and split it
        # Overwrite particles in the dimensional PSO with the split vecs
        split_gbest_vec = pso2.split_gbest_vec(vec=pso1.gbest_pos)
        for pso2, gbest_vec in zip(psos2, split_gbest_vec):
            
            pso2.overwrite_particle(vec=gbest_vec)


        gbest_val_list.append(pso1.gbest_val)

        print(f"Iteration: {i} gbest_val: {round(pso1.gbest_val,2)}")

        for pso2 in psos2:
            # pso2.set_constricted_velocity()
            # pso2.move_random_back()
            pso2.mix()
            pso2.move()
        
        # pso1.set_constricted_velocity()
        # pso1.move_random_back()
        pso1.mix()
        pso1.move()

    
    end = time.perf_counter()
    total_time = end-start
    
    print("")
    print(f'{iterations} Iterations took {total_time}')
        
    return gbest_val_list, total_time

        

if __name__ == '__main__':
    
    gbest_vals, total_time = main()
    plot_results(gbest_vals, total_time)

    

    



    



