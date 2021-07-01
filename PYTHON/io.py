'''
File for file preparation and Mol2Reader class
'''

from pymatgen.io.cif import CifParser
from pymatgen.io.xyz import XYZ
from openbabel import openbabel
import numpy as np
import pandas as pd 
import networkx as nx
from .Atom import Atom
from .Bond import Bond
from .Molecule import Molecule
from .Interaction import *
import os

def Cif2Supercell(input_path,supercell_size,occupancy_tolerance=1, output_filename='supercell',output_path='.'):
    # Read in Cif file and create supercell. Save as XYZ file
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    read_cif = CifParser(input_path,occupancy_tolerance=occupancy_tolerance)
    struc = read_cif.get_structures()[0]
    struc.make_supercell(supercell_size, to_unit_cell=False)
    xyzrep = XYZ(struc)
    xyzrep.write_file(f"{output_path}/{output_filename}.xyz")  # write supercell to file
    # Convert supercell to Mol2 format
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol2")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, f"{output_path}/{output_filename}.xyz")   # Open Babel will uncompress automatically
    mol.AddHydrogens()
    obConversion.WriteFile(mol, f'{output_path}/{output_filename}.mol2')

class Mol2Reader():
    def __init__(self,path):
        self.path = path 

    def read(self):
        file = open(self.path).readlines()
        tripos = ''
        g = nx.Graph()
        atom_count = 1
        # Parse the mol2 file
        for line in file:
            if '@<TRIPOS>' in line:
                if 'ATOM' in line:
                    tripos = 'atom'
                    continue
                elif 'BOND' in line:
                    tripos = 'bond' 
                    continue
                else:
                    tripos = -1
                    continue
            if tripos == 'atom':
                this_line = self.line_to_list(line)
                atom_node = Atom(atom_symbol=this_line[1],atom_type=this_line[5],
                                                    coordinates=np.array([this_line[2],this_line[3],this_line[4]]),
                                                    label=f'{this_line[1]}{this_line[0]}')
                g.add_node(atom_count,data=atom_node)
                atom_count += 1
            if tripos == 'bond':
                this_line = self.line_to_list(line)
                bond_edge = Bond(atom1=g.nodes[this_line[1]]['data'],atom2=g.nodes[this_line[2]]['data'],order=this_line[3])
                # Add bonds and neighbours to atoms
                g.nodes[this_line[1]]['data'].bonds.append(bond_edge)
                g.nodes[this_line[1]]['data'].neighbours.append(g.nodes[this_line[2]]['data'])
                g.nodes[this_line[2]]['data'].bonds.append(bond_edge)
                g.nodes[this_line[2]]['data'].neighbours.append(g.nodes[this_line[1]]['data'])
                g.add_edge(this_line[1],this_line[2],data=bond_edge)
        # Retrieve subgraphs
        subgraphs = [g.subgraph(c) for c in nx.connected_components(g)] 
        max_nodes = max([len(subgraph.nodes) for subgraph in subgraphs])
        subgraphs = [subgraph for subgraph in subgraphs if len(subgraph.nodes) == max_nodes]
        molecules = []
        for graph in subgraphs:
            molecules.append(Molecule([node[1]['data'] for node in graph.nodes(data=True)],
                                      [edge[2]['data'] for edge in graph.edges(data=True)]))
        return molecules

    def line_to_list(self,line):
        line = line.replace(' ',',')
        line+=(',')
        line_list = []
        temp_string = ''
        for char in line:
            if ((char == ',') & (temp_string != '')):
                try:
                    if '.' in temp_string:
                        line_list.append(np.float(temp_string))
                        temp_string = ''
                    else:
                        line_list.append(np.int(temp_string))
                        temp_string = ''
                except:
                    line_list.append(temp_string)
                    temp_string = ''
            else:
                temp_string += char if char != ',' else ''
        return line_list