import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass, field
from typing import List



@dataclass
class PSO:
    num_particles: int = 20
    particles: List = field(init=False)
    gbest_val: float = np.Inf
    gbest_pos: np.array = field(init=False)


    def __post_init__(self):
        self.particles = [dict() for _ in range(self.num_particles)]


    def initialise(self):
        for particle in self.particles:
            particle['position'] = random_initiate_vec()
            particle['pbest_val'] = np.Inf
            particle['velocity'] = np.zeros(particle['position'].size)
    

    def initialise_with_particle_list(self, particle_pos_list):
        for particle, pos in zip(self.particles, particle_pos_list):
            particle['position'] = pos
            particle['pbest_val'] = np.Inf
            particle['velocity'] = np.zeros(particle['position'].size)
    

    def pick_informants_ring_topology(self):
        for index, particle in enumerate(self.particles):
            particle['informants'] = []
            particle['lbest_val'] = np.Inf
            particle['informants'].append(self.particles[(index-1) % len(self.particles)])
            particle['informants'].append(self.particles[index])
            particle['informants'].append(self.particles[(index+1) % len(self.particles)])
    

    def calculate_fitness(self):
        for particle in self.particles:
            particle['error'] = calculate_error(particle['position'], sup_cha=rs)

    
    def set_pbest(self):
        for particle in self.particles:
            if particle['error'] < particle['pbest_val']:
                particle['pbest_val'] = particle['error']
                particle['pbest_pos'] = particle['position']
    

    def set_lbest(self):
        for particle in self.particles:
            for informant in particle['informants']:
                if(informant['pbest_val'] <= particle['lbest_val']):
                    informant['lbest_val'] = particle['pbest_val']
                    informant['lbest_pos'] = particle['pbest_pos']
    
    
    def set_gbest(self):
        for particle in self.particles:
            if particle['lbest_val'] <= self.gbest_val:
                self.gbest_val = particle['lbest_val']
                self.gbest_pos = particle['lbest_pos']
    
    
