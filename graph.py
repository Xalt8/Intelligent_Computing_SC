import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, field, InitVar


@dataclass
class SupplyChain():
    graph: nx.DiGraph = field(init=False)
    farms: dict
    customers: dict
    products: dict
    demand: dict
    prices: dict
    transport: dict
    dealer: str = 'Dealer'
    

    def __repr__(self) -> str:
        return f'SupplyChain(farms:{len(self.farms.keys())} customers:{len(self.customers.keys())} products:{len(self.products.keys())})'


    def set_supply_quantity(self) -> None:
        ''' Takes values from dict(self.farms) and puts it into the graph as edges'''
        for farm in self.farms.keys():
            self.graph[farm][self.dealer]['qty_supplied']=self.farms[farm]['Qty']
    

    def set_supply_price(self) -> None:
        ''' Takes a dict with supply vales and puts it into the graph as edges'''
        for farm in self.farms.keys():
            self.graph[farm][self.dealer]['price_supplied']=self.farms[farm]['Cost']
    

    def set_demand_quantity(self)-> None:
        for product in self.products.keys():
            for customer in self.customers.keys():
                self.graph[product][customer]['demand'] = self.demand[product][customer]


    def set_prices(self) -> None:
        for product in self.products.keys():
            for customer in self.customers.keys():
                self.graph[product][customer]['price'] = self.prices[product][customer]


    def __post_init__(self):
        self.graph = nx.DiGraph()
        
        self.graph.add_nodes_from([(k,v) for k,v in self.farms.items()])
        self.graph.add_nodes_from([(k,v) for k,v in self.customers.items()])
        self.graph.add_nodes_from([(k,{'eggs_per_box':v}) for k,v  in self.products.items()])
        self.graph.add_node(self.dealer)
        
        self.graph.add_edges_from([(farm, self.dealer) for farm in self.farms.keys()])
        self.graph.add_edges_from([(self.dealer, product) for product in self.products.keys()])
        self.graph.add_edges_from([(product, customer) for product in self.products.keys() for customer in self.customers.keys()])

        self.set_supply_quantity()
        self.set_supply_price()

        self.set_demand_quantity()
        self.set_prices()
            
    
    def get_demand_vec(self) -> np.ndarray:
        ''' Returns an array with demand quantities'''
        return np.array([self.graph[p][cust]['demand'] for p in self.products for cust in self.graph.successors(p)]).astype(np.int64)

    
    def get_total_supply(self) -> int:
        ''' Return the total number of eggs supplied by farms to dealer'''
        return np.sum([self.graph[farm]['Dealer']['qty_supplied'] for farm in self.graph.predecessors('Dealer')])

    
    def get_supply_costs(self, vec: np.ndarray)-> float:
        ''' Gets the total cost of eggs supplied given a vector of product quantities'''
        avg_cost_per_egg = np.mean([self.graph[farm]['Dealer']['price_supplied'] for farm in self.graph.predecessors(self.dealer)])
        mat = vec.reshape(len(self.products), len(self.customers))
        prod_cap = np.expand_dims(np.array([self.graph.nodes[p]['eggs_per_box'] for p in self.products]),axis=1)
        total_eggs = np.sum(mat * prod_cap)
        return total_eggs * avg_cost_per_egg


    def get_transport_cost(self, vec:np.ndarray) -> float:
        ''' Returns the total transport cost for a given array of quantities '''
        dims = (len(self.products), len(self.customers))
        per_egg_mat = np.array([self.transport[self.graph.nodes[c]['Location']] 
                        for p in self.products for c in self.graph.successors(p)]
                        ).reshape(dims)
        mat = vec.reshape(dims)
        prod_cap = np.expand_dims(np.array([self.graph.nodes[p]['eggs_per_box'] for p in self.products]),axis=1)
        return np.sum(mat * per_egg_mat * prod_cap)
    
    
    def get_price_per_product(self) -> np.ndarray:
        ''' Returns the prices for the products '''
        return np.array([self.graph[p][c]['price'] for p in self.products for c in self.graph.successors(p)])


if __name__ =='__main__':
    pass