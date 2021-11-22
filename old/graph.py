import numpy as np
import networkx as nx
from dataclasses import dataclass, field




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


    def get_demand_by_product(self, product:str) -> np.ndarray:
        return np.array([self.graph[p][ c]['demand'] for p in  self.products for c in self.graph.successors(p) if p==product]) 


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

    
    def get_supply_costs_by_product(self, vec:np.ndarray, product:str)-> float:
        avg_cost_per_egg = np.mean([self.graph[farm]['Dealer']['price_supplied'] for farm in self.graph.predecessors(self.dealer)])
        total_eggs = np.sum(vec * self.products[product])
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
    

    def get_transport_cost_by_product(self, vec:np.ndarray, product:str) -> float:
        per_egg = np.array([self.transport[self.graph.nodes[c]['Location']] 
                    for p in self.products for c in self.graph.successors(p) if p==product])
        eggs_per_box = self.products[product]
        return np.sum(vec * per_egg * eggs_per_box)
    

    def get_price_per_product(self) -> np.ndarray:
        ''' Returns the prices for the products '''
        return np.array([self.graph[p][c]['price'] for p in self.products for c in self.graph.successors(p)])


    def get_boxes(self, vec: np.ndarray) -> np.ndarray:
        ''' Returns an array with boxes of products '''
        return np.sum(vec.reshape(len(self.products), len(self.customers)), axis=1)


    def get_supplied_eggs(self, vec: np.ndarray) -> int:
        ''' Returns the number of eggs supplied to customers based on 
            quantity of boxes (vec) '''
        prod_cap = np.array([self.graph.nodes[p]['eggs_per_box'] for p in self.products])
        boxes = self.get_boxes(vec)
        return np.sum([np.sum(box * pc) for box, pc in zip(boxes, prod_cap)])


    def allocate_supply_by_product(self) -> dict:
        ''' Allocates the available eggs supply in the same propotion as demand'''
        dem_vec = self.get_demand_vec()
        dem_boxes = self.get_boxes(dem_vec) # boxes
        total_supply = self.get_total_supply() # eggs
        eggs_qty = [int(total_supply*(db/np.sum(dem_boxes))) for db in dem_boxes]
        return {product:eggs for product, eggs in zip(self.products.keys(), eggs_qty)}


    def get_ordered_indices(self) -> np.ndarray:
        '''Retuns a list of index positions with the largest customers by demand first'''
        dem_mat = np.array([
            self.graph[p][c]['demand'] for p in self.products for c in self.graph.successors(p) 
        ]).reshape(len(self.products), len(self.customers))
        
        prod_cap = np.expand_dims(np.array([self.graph.nodes[p]['eggs_per_box'] for p in self.products]), axis=1)
        total_dem_eggs = np.sum(dem_mat * prod_cap, axis=0).astype(np.int64)
        return np.argsort(total_dem_eggs)[::-1]

    
    def get_big_customer_indices(self) -> np.ndarray:
        ''' Retuns a list of index positions with the largest customers by demand in 
            descending order'''
        dem_mat = np.array([
            self.graph[p][c]['demand'] for p in self.products for c in self.graph.successors(p) 
        ]).reshape(len(self.products), len(self.customers))
        
        prod_cap = np.expand_dims(np.array([self.graph.nodes[p]['eggs_per_box'] for p in self.products]), axis=1)
        total_dem_eggs = np.sum(dem_mat * prod_cap, axis=0).astype(np.int64)
        return np.argsort(total_dem_eggs)[::-1]


    def put_max_val(self, index:int, vec:np.ndarray) -> np.int64:
        ''' Puts the maximum possible value given demand and supply constraints for a
            given index position in a vector '''
        demand = self.get_demand_vec()
        supply = self.get_total_supply()
        prod_cap = np.array([self.graph.nodes[p]['eggs_per_box'] for p in self.products])

        dims = (len(self.products), len(self.customers))

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
                return np.minimum(available_supply_boxes.item(), demand[index].item())
            else:
                return 0
                

    def fill_order(self) -> np.ndarray:
        ''' Fill order of largers customers first '''
        index_order = self.get_big_customer_indices() # -> customer indices of largest customers
        vec = np.zeros(len(self.products) * len(self.customers))
        for io in index_order:
            for i in range(len(self.products)):
                vec[io + i*len(self.customers)] = self.put_max_val(index = io + i*len(self.customers), vec=vec)
        return vec


    def get_global_variables(self) -> tuple:
        '''Used to set up global variables for optimisation'''
        demand = self.get_demand_vec()
        supply = self.get_total_supply()
        dims = (len(self.products), len(self.customers))
        prod_cap = np.array([self.graph.nodes[p]['eggs_per_box'] for p in self.products])
        return demand, supply, dims, prod_cap


if __name__ =='__main__':
    pass