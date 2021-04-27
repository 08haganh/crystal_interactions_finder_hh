# Testing Crystal Interactions Finder HH with example script
import sys 
sys.path.append('/home/harry/Documents/PhD/crystal_interactions_finder_hh')
from PYTHON.io import *
from PYTHON.Atom import *
from PYTHON.Bond import * 
from PYTHON.Molecule import *
from PYTHON.Crystal import * 
from PYTHON.Interaction import *
from PYTHON.Geometry import * 
from PYTHON.Viz import *

import pandas as pd 
import numpy as np
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from datetime import datetime

''' 
This example script creates a 5*5*5 supercell from a chosen CIF and calculates: 
1) Atomic distances between atoms in the central molecule and atoms in every other molecule
2) The geometric relations between all pairs of molecules in the crystal e.g. displacement, distance, inteprlanar angle
3) The supramolecular interactions present in the central molecule to its neighbours
4) Saves the atomic interactions and molecular interactions as a csv file

This script can be incorporated into a for loop and run on a large dataset. 
You may want to tweak the supercell size to 4*4*4 for time considerations
'''

start = datetime.now()

# Generate Supercell from cif
Cif2Supercell('CEKGAB.cif',supercell_size=[[4,0,0],[0,4,0],[0,0,4]],output_filename='supercell',output_path='.')
# Read in mol2 and generate list of Molecule objects
mol2_reader = Mol2Reader('supercell.mol2')
mols = mol2_reader.read()
# As we are working with acenes and aromatic rings we need to do some preprocessing
acenes = []
for mol in mols:
    mol.add_rings_as_atoms()
    mol.assign_atoms_interaction_dict()
    acenes.append(Acene(mol))
crystal = Crystal(acenes)
# Atom Distances from central molecule to all other molecules in the Crystal
central_molecule, central_idx = crystal.get_central_molecule(return_idx=True)
central_cog = central_molecule.centre_of_geometry()
central_atom_coords = np.array([atom.coordinates for atom in central_molecule.atoms]) # shape = (n_atoms,3)
all_atom_coords = []
for mol in crystal.molecules:
    all_atom_coords.append(np.array([atom.coordinates for atom in mol.atoms]))
all_atom_coords = np.array(all_atom_coords) # shape = (n_mols,n_atoms,3) 
dists = []
mol1s = []
mol2s = []
for i, mol_coords in enumerate(all_atom_coords): # Distances are done on batch between molecule i atom j and molecule k atom j + x instead of 1 by 1
    temp_dist = []
    for x in range(len(mol_coords)):
        mol1s += [central_idx]*len(mol_coords)
        mol2s += [i]*len(mol_coords)
        disp = mol_coords - central_atom_coords # shape = (n_atoms,3)
        dist2 = disp[:,0] * disp[:,0] + disp[:,1] * disp[:,1] + disp[:,2] * disp[:,2]
        dist = np.sqrt(dist2) # shape = (n_atoms)
        temp_dist.append(dist)
        mol_coords = np.roll(mol_coords,-1,axis=0)
    dists.append(temp_dist)
dists = np.array(dists) # shape = (n_molecules,x_atoms,y_atoms) | where y in y_atoms = dist(atom_x_central - atom_y_mol_n)
## Organise distances with molecule and atom index for good book-keeping and easy reference later
in_atom_order = np.array([dist.flatten('F') for dist in dists]).reshape(-1)
d1 = dists.shape[0]
d2 = dists.shape[1]
arange = np.arange(d2)
atom1s = np.concatenate([[x]*d2 for x in range(d2)]*d1)
atom2s = np.concatenate([np.roll(arange,-x) for x in range(d2)]*d1)
## Turn Atom Distances to DataFrame
data_dict= {'mol1s':mol1s,'mol2s':mol2s,'atom1s':atom1s,'atom2s':atom2s,'dists':in_atom_order}
atom_dist_df = pd.DataFrame(data_dict)
atom_dist_df = atom_dist_df[atom_dist_df.mol1s != atom_dist_df.mol2s]
# Looping through geometric relations doesnt take long so can use a nested for loop
all_mols = []
for mol in acenes:
    all_mols.append(mol.centre_of_geometry())
all_mols = np.array(all_mols)
cog_disps = []
cog_dists = []
cog_mol1s = []
cog_mol2s = []
interplanar_angles = []
unit_cell_disps = []
planes = []
## Create a list of molecular planes
for mol in crystal.molecules:
    planes.append(Plane(mol.get_backbone().atoms))
## Loop through all pairs of molecules
for i, arr1 in enumerate(all_mols[:-1]):
    for j, arr2 in enumerate(all_mols[i+1:],i+1):
        interplanar_angles.append((planes[i].plane_angle(planes[j])))
        cog_mol1s.append(i)
        cog_mol2s.append(j)
        disp = arr2 - arr1
        unit_cell_disps.append(disp)
        dist = np.sqrt(disp.dot(disp))
        cog_disps.append(disp)
        cog_dists.append(dist)
## Turn lists to arrays      
unit_cell_disps = np.array(unit_cell_disps)        
cog_dists = np.array(cog_dists)
## Create Molecule Geometry to DataFrame
data_dict= {'mol1s':cog_mol1s,'mol2s':cog_mol2s,
            'a':unit_cell_disps[:,0],'b':unit_cell_disps[:,1],'c':unit_cell_disps[:,2],
            'dists':cog_dists,'interplanar_angles':interplanar_angles}
df_cogs = np.round(pd.DataFrame(data_dict).set_index(['mol1s','mol2s']),3)
# Calculate Interactions for atom pairs whose distance is less than 6A
atom_contacts = []
red_df = atom_dist_df.loc[atom_dist_df.dists < 6]
for idx in red_df.index:
    mol1_idx = red_df.at[idx,'mol1s']
    mol2_idx = red_df.at[idx,'mol2s']
    atom1_idx = red_df.at[idx,'atom1s']
    atom2_idx = red_df.at[idx,'atom2s']
    atom1 = crystal.molecules[mol1_idx].atoms[atom1_idx]
    atom2 = crystal.molecules[mol2_idx].atoms[atom2_idx]
    coords1 = atom1.coordinates
    coords2 = atom2.coordinates
    interaction = Interaction(atom1,atom2)
    disp = coords2 - coords1
    atom_contact = {'mol1s':mol1_idx,'mol2s':mol2_idx,'atom1s':atom1_idx,'atom2s':atom2_idx,
                    'a':disp[0],'b':disp[1],'c':disp[2]}
    atom_contact.update(interaction.to_dict())
    atom_contacts.append(atom_contact)
ac_df = pd.DataFrame(atom_contacts).set_index(['mol1s','mol2s'])
# sum atomic interactions for each molecular pair and add to geometric relations dataframe
hbond_bond = pd.DataFrame(ac_df.groupby(ac_df.index)['hydrogen_bond'].sum()).set_index(ac_df.index.unique())
pi_pi_bond = pd.DataFrame(ac_df.groupby(ac_df.index)['pi_pi_bond'].sum()).set_index(ac_df.index.unique())
halogen_bond = pd.DataFrame(ac_df.groupby(ac_df.index)['halogen_bond'].sum()).set_index(ac_df.index.unique())
ch_pi_bond = pd.DataFrame(ac_df.groupby(ac_df.index)['ch_pi_bond'].sum()).set_index(ac_df.index.unique())
hydrophobic_bond = pd.DataFrame(ac_df.groupby(ac_df.index)['hydrophobic_cc_bond'].sum()).set_index(ac_df.index.unique())
vdw_contact = pd.DataFrame(ac_df.groupby(ac_df.index)['vdw_contact'].sum()).set_index(ac_df.index.unique())
fin_df = pd.concat([df_cogs,vdw_contact,hbond_bond,pi_pi_bond,halogen_bond,ch_pi_bond,
                    hydrophobic_bond],axis=1)
ac_df.to_csv(f'atom_interactions_central_molecule.csv',index=True)
# Align interactions properly to remove double counting of indices
double_counted = fin_df.loc[fin_df.index.get_level_values(0) > fin_df.index.get_level_values(1)]
double_counted = double_counted[['vdw_contact','hydrogen_bond','pi_pi_bond','halogen_bond',
                                    'ch_pi_bond','hydrophobic_cc_bond']]
fin_df.drop(double_counted.index,axis=0,inplace=True)
arrays = [double_counted.index.get_level_values(1),double_counted.index.get_level_values(0)]
tuples = list(zip(*arrays))
double_counted.index = pd.MultiIndex.from_tuples(tuples, names=["mol1s", "mol2s"])
fin_df.loc[double_counted.index,double_counted.columns] = double_counted
fin_df.to_csv(f'molecule_interactions_central_molecule.csv',index=True)

# Draw crystal graph of central molecule
g = create_crystal_graph_central(fin_df.reset_index(),consider_interactions='all')
network_plot_3D(g, 0, save=False)

# Draw the complete crystal graph
g = create_crystal_graph(fin_df.reset_index(),consider_interactions='all')
network_plot_3D(g, 0, save=False)