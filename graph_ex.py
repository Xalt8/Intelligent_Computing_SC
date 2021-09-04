import numpy as np
import networkx as nx
from dataclasses import dataclass
import pandas as pd


xls = pd.ExcelFile('data_ex.xlsx')
demand_dict = pd.read_excel(xls, sheet_name='demand', index_col=0, usecols="A:D", nrows=33).to_dict('index')  
prices_dict = pd.read_excel(xls, sheet_name = 'prices', index_col=0, usecols="A:D", nrows=33).to_dict('index') 
custs = pd.read_excel(xls, sheet_name='customers', index_col=0, usecols="A:C", nrows=33).to_dict('index')
farms_dict = pd.read_excel(xls, sheet_name='farms', index_col=0, usecols="A:F", nrows=31).to_dict('index')
products_dict = {'P1': 6, 'P2': 10, 'P3': 12}
transport_cost_per_egg = {'same_loc':0.10, 'different_loc':0.15}


def sum_to_num(sum_to, nums):
    ''' Generates random numbers that sum up to a given value
        Parameters:
        ----------
        sum_to: The values the numbers should sum up to.
        nums: The number of random numbers.
        Source: http://sunny.today/generate-random-integers-with-fixed-sum/
    '''
    return np.random.multinomial(sum_to, np.ones(nums)/nums, size=1)[0]



@dataclass
class SupplyChain():
    farms: dict
    customers: dict
    products: dict
    demand: dict
    prices: dict
    transport: dict
    graph: nx.DiGraph = nx.DiGraph()

    
    def edges_fprod_cprod(self) -> list:
        ''' Creates tuples of customers-products &
            farm-products for edges '''
        prod_edges = []
        for customer in self.customers.keys():
            for product in self.products.keys():
                for farm in self.farms.keys():
                    for f_prod in self.farms[farm]['Products']:
                        if product == f_prod:
                            prod_edges.append((farm + "_" + f_prod, customer + "_" + product))
        return prod_edges
    
    def set_demand_quantity(self): 
        ''' Adds node attribute -> demand and fills it with demand value from dict'''
        for customer in self.customers.keys():
            for product in self.products.keys():
                self.graph.nodes[customer + "_" + product]['demand'] = self.demand[customer][product]

    # def set_demand_quantity(self):
    #     ''' Set the demand quantity on the edge between cprod and customer node'''
    #     for customer in self.customers.keys():
    #         for product in self.products.keys():
    #             self.graph[customer+"_"+product][customer]['demand'] = self.demand[customer][product]

    # Buy prices on nodes
    # def set_buying_prices(self):
    #     ''' Adds node attribute -> price and fills it with price values from dict'''
    #     for customer in self.customers.keys():
    #         for product in self.products.keys():
    #             self.graph.nodes[customer + "_" + product]['price'] = self.prices[customer][product]
    
    def set_buying_prices(self):
        ''' Set the buying price on the edge between cprod and customer node'''
        for customer in self.customers.keys():
            for product in self.products.keys():
                self.graph[customer+"_"+product][customer]['price'] = self.prices[customer][product]


    def set_supply_quantity(self):
        ''' Adds eggs_supply node attribute for farms and fills it with farm Qty values from dict'''
        for farm in self.farms.keys():
            self.graph.nodes[farm]['eggs_supply'] = self.farms[farm]['Qty']
    
    # def set_supply_price(self):
    #     ''' Adds cost_per_egg node attribute for farms and fills it with Cost values from dict '''
    #     for farm in self.farms.keys():
    #         self.graph.nodes[farm]['cost_per_egg'] = self.farms[farm]['Cost']

    def set_supply_cost(self):
        ''' Adds the cost_per_egg for edge between farm and fprods'''
        for farm in self.farms.keys():
            for fprod in self.graph.successors(farm):
                for cprod in self.graph.successors(fprod):
                    self.graph[fprod][cprod]['cost_per_egg'] = self.farms[farm]['Cost']        

    def set_fprod_locations(self):
        ''' Sets the location of the f_prod to the farm location '''
        for farm in self.farms.keys():
            for product in self.farms[farm]['Products']:
                self.graph.nodes[farm + "_" + product]['Location'] = self.farms[farm]['Location']

    def set_cprod_locations(self):
        for cust in self.customers.keys():
            for prod in self.products.keys():
                self.graph.nodes[cust + "_" + prod]['Location'] = self.customers[cust]['Location']

    def set_transport_costs(self):
        ''' Set the transport costs on the edges between fprod and cprod based on their locations'''
        # f_prods = [farm + "_" + product for farm in self.farms.keys() for product in self.farms[farm]['Products']]
        for farm in self.farms.keys():
            for fprod in self.graph.successors(farm):
                for cprod in self.graph.successors(fprod):
                    if self.graph.nodes[fprod]['Location'] == self.graph.nodes[cprod]['Location']:
                        self.graph[fprod][cprod]['transport_cost'] = self.transport['same_loc']
                    else:
                        self.graph[fprod][cprod]['transport_cost'] = self.transport['different_loc']
                        
    def set_eggs_packs(self):
        ''' Sets the eggs per pack on fprod and farm edge'''
        for farm in self.farms.keys():
            for product in self.farms[farm]['Products']:
                self.graph.nodes[farm + "_" + product]['eggs_per_pack'] = self.products[product]



    def __post_init__(self):

        # Fix the products list
        for farm in self.farms.keys():
            self.farms[farm]['Products'] = self.farms[farm]['Products'].split(', ')
        # Add farm nodes and connect to f_prods
        self.graph.add_edges_from([(farm, farm + "_" + product) for farm in self.farms.keys() 
                                                                for product in self.farms[farm]['Products']])      
        # Add customer nodes and connect to c_prods
        self.graph.add_edges_from([(cust + "_" + prod, cust) for cust in self.customers.keys() 
                                                            for prod in self.products.keys()])
        # connect f_prod & c_prod
        self.graph.add_edges_from(self.edges_fprod_cprod())

        self.set_demand_quantity()
        self.set_buying_prices()
        self.set_supply_quantity()
        self.set_supply_cost()
        self.set_fprod_locations()
        self.set_cprod_locations()
        self.set_transport_costs()
        self.set_eggs_packs()

    # Get demand values from nodes
    # def get_demand_vec(self) -> np.ndarray:
    #     ''' Gets the demand node attributes from the graph as a dict 
    #         converts the dict's values to an array'''
    #     return np.fromiter(nx.get_node_attributes(self.graph,'demand').values(), dtype=np.int64)

    # Get demand value from edges
    def get_demand_vec(self) -> np.ndarray:
        ''' Returns an array with the demand values between cprod and customer edges'''
        return np.array([self.graph[cprod][customer]['demand'] 
        for customer in self.customers.keys() 
        for cprod in self.graph.predecessors(customer)]).astype(np.int64)
    

    def get_price_per_product(self) -> np.ndarray:
        return np.fromiter(nx.get_node_attributes(self.graph,'price').values(), dtype=np.float64)

    def get_eggs_supplied(self) -> np.ndarray:
        return np.fromiter(nx.get_node_attributes(self.graph,'eggs_supply').values(), dtype=np.int64)

    # def get_supply_costs(self) -> np.ndarray:
    #     ''' Gets the cost per egg supplied from farm to customer'''
    #     costs = []
    #     for farm in self.farms.keys():
    #         for prod in self.farms[farm]['Products']:
    #             fprod = farm + "_" + prod
    #             for _ in self.graph.successors(fprod):
    #                 costs.append(self.farms[farm]['Cost'])
    #     return np.array(costs).astype(np.float16)
    
    def get_supply_costs(self) -> np.ndarray:
        ''' Gets the cost per egg supplied from fprod to cprod'''
        supply_costs =[]
        for farm in self.farms.keys():
            for fprod in self.graph.successors(farm):
                for cprod in self.graph.successors(fprod):
                    supply_costs.append(self.graph[fprod][cprod]['cost_per_egg'])
        return np.array(supply_costs).astype(np.float64)

    # def get_transport_cost(self):
    #     ''' Gets the transport costs per egg supplied from farm to customer'''
    #     costs =[]
    #     for farm in self.farms.keys():
    #         for prod in self.farms[farm]['Products']:
    #             fprod = farm + "_" + prod
    #             for cust_prod in self.graph.successors(fprod):
    #                 if self.graph.nodes[cust_prod]['Location'] == self.graph.nodes['F1_P1']['Location']:
    #                     transport_cost = self.transport['same_loc']
    #                 else:
    #                     transport_cost = self.transport['different_loc']
    #                 costs.append(transport_cost)
    #     return np.array(costs).astype(np.float16)
    
    def get_transport_cost(self):
        ''' Returns an array with the transport costs of the fprod to cprod'''
        transport_costs = []
        for farm in self.farms.keys():
            for fprod in self.graph.successors(farm):
                for cprod in self.graph.successors(fprod):
                    transport_costs.append(self.graph[fprod][cprod]['transport_cost'])
        return np.array(transport_costs).astype(np.float16)


    def split_eggs_supply_randomly(self) -> list:
        ''' Retuns a list of arrays with the total quantity supplied distributed among the total number of products '''
        dims = np.sum([len(self.farms[farm]['Products']) for farm in self.farms.keys()])
        z_vec = np.zeros(dims)
        split_indices = np.cumsum([len(self.farms[farm]['Products']) for farm in self.farms.keys()])
        split_vec = np.split(z_vec, split_indices)[:-1]  # Pop the last one -> its empty
        return [sum_to_num(supply, len(sv)) for sv, supply in zip(split_vec, self.get_eggs_supplied())]

    def pack_random_eggs(self) -> list:
        packs = []
        fprods = [self.farms[farm]['Products'] for farm in self.farms.keys()]
        for fprod, randegg in zip(fprods, self.split_eggs_supply_randomly()):
            for fp, re in zip(fprod, randegg):
                packs.append(np.floor(re/self.products[fp]))
        return np.array(packs).astype(np.int64)

if __name__ =='__main__':
    
    sc = SupplyChain(farms=farms_dict, customers=custs, 
    products=products_dict, demand=demand_dict, 
    prices=prices_dict, transport=transport_cost_per_egg)

