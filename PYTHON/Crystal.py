'''
File to store the Crystal class
Attributes
- molecules; list; list of all of the molecule objects in the crystal
Methods
- add_molecule(Molecule); return None; appends a Molecule object to molecules list
- add_molecules(list); return None; iterates through list of Molecule objects and appends them to molecules list
- centre_of_geometry(); return np.array; returns the coordinates of the centre of geometry of the crystal as a numpy array
- get_intermolecular_interactions(); return list; returns a list of Interaction objects for each pair of atoms that 
do not share a parent molecule in the crystal
- get_centroid_displacements(basis=None); return 
- get_central_molecule(); return Molecule; returns the molecule that is closest to the centre of geometry of the crystal
- get_molecule_centroids(); return list; returns a list of np.array objects of the coordinates of the centre of geometry
for each molecule in the crystal
- get_unique_dimers(); return list; returns a list of Molecule objects containing the unique dimers in the crystal
- get_molecule_atom_distances(mol1_index,mol2_index); return list; returns list of intermolecular atomic distances between two
molecules in the crystal
- get_molecule_atom_vdw_distances(mol1_index,mol2_index); return list; returns list of intermolecular atomic distances minus
the sum of their vdw_radii between two molecules in the crystal
- to_nx_graph(by='all'/'molecular_centroids'); return Networkx graph object; returns an nx graph object of the crystal.
If by ='all' the full list of atoms are the nodes and the covalent and intermolecular_bonds are the edges
If by='molecular_centroids' the molecular centroids are the nodes and the edges are the set of intermolecular interactions between
two molecules. default = 'all'
'''
from .Atom import Atom
from .Bond import Bond
from .Molecule import Molecule, Acene
from .Interaction import *
from .Geometry import *
import numpy as np
import os
import shutil
from openbabel import openbabel
import pandas as pd

def calc_lstsq_displacement(disp,vectors):
    A = vectors.T
    xs = []
    x, _, _, _ = np.linalg.lstsq(A,disp,rcond=-1)
    xs.append(x)
    return np.array(xs[0])

class Crystal():
    def __init__(self,molecules=[]):
        self.molecules = molecules
        self.dimers = []

    def add_molecule(self,molecule):
        self.molecules.append(molecule)

    def add_molecules(self,molecules):
        for molecule in molecules:
            self.add_molecule(molecule)

    def centre_of_geometry(self):
        mol_centroids = self.get_molecule_centroids()
        return np.mean(mol_centroids,axis=0)
    

    def to_xyz(self,filename):
        atom_symbols = np.array([atom.symbol for molecule in self.molecules for atom in molecule.atoms])
        atoms = [atom for molecule in self.molecules for atom in molecule.atoms]
        unique_atoms = np.unique(atom_symbols)
        atom_count_dict = {}
        for unique_atom in unique_atoms:
            atom_count_dict[unique_atom] = np.sum(np.isin(atom_symbols,unique_atom))
        with open(f'{filename}.xyz','w') as file:
            file.write(f'{len(atom_symbols)}\n')
            for key in atom_count_dict.keys():
                file.write(f'{key}{atom_count_dict[key]} ')
            file.write('\n')
            for atom in atoms:
                coords = atom.coordinates
                file.write(f'{atom.symbol} {coords[0]} {coords[1]} {coords[2]}\n')
        return None

    def get_central_molecule(self,return_idx=False):
        crystal_cog = np.array(self.centre_of_geometry())
        mol_cogs = np.array(self.get_molecule_centroids())
        displacements = np.array([mol_cog - crystal_cog for mol_cog in mol_cogs])
        distances = [np.sqrt(displacement.dot(displacement)) for displacement in displacements]
        idxs = [x for x in range(len(self.molecules))]
        idx = idxs[np.where(distances == np.min(distances))[0][0]]
        if return_idx:
            return self.molecules[idx], idx
        else:
            return self.molecules[idx]

    def get_molecule_centroids(self):
        mol_centroids = []
        for molecule in self.molecules:
            mol_centroids.append(molecule.centre_of_geometry())
        return mol_centroids

    def unique_dimers_to_xyz(self):
        com_distances = []
        for i, mol1 in enumerate(self.molecules):
            for j, mol2 in enumerate(self.molecules[i+1:],i+1):
                cog1 = mol1.centre_of_geometry()
                cog2 = mol2.centre_of_geometry()
                displacement = cog2 - cog1 
                distance = np.round(np.sqrt(displacement.dot(displacement)),3)
                atom_distances = self.get_molecule_atom_distances(i,j)
                if distance in com_distances:
                    continue
                elif ((distance > 5) & (np.min(atom_distances) > 5)):
                    continue
                else:
                    dimer = Molecule(atoms = mol1.atoms+mol2.atoms,
                                     bonds = mol1.bonds+mol2.bonds)
                    dimer.to_xyz(f'mol{i}_mol{j}_dimer')
                com_distances.append(distance)

    def get_unique_dimers(self):
        dimers = []
        com_distances = []
        for i, mol1 in enumerate(self.molecules):
            for j, mol2 in enumerate(self.molecules[i+1:],i+1):
                cog1 = mol1.centre_of_geometry()
                cog2 = mol2.centre_of_geometry()
                displacement = cog2 - cog1 
                distance = np.round(np.sqrt(displacement.dot(displacement)),3)
                atom_distances = self.get_molecule_atom_distances(i,j)
                if distance in com_distances:
                    continue
                elif ((distance > 5) & (np.min(atom_distances) > 5)):
                    continue
                else:
                    dimer = Molecule(atoms = mol1.atoms+mol2.atoms,
                                     bonds = mol1.bonds+mol2.bonds)
                    dimers.append(dimer)
                com_distances.append(distance)
        return dimers 
        
    def unique_dimers_to_mol(self):
        os.mkdir('./tempdir')
        os.chdir('./tempdir')
        file = open('names.txt','w')
        com_distances = []
        for i, mol1 in enumerate(self.molecules):
            for j, mol2 in enumerate(self.molecules[i+1:],i+1):
                cog1 = mol1.centre_of_geometry()
                cog2 = mol2.centre_of_geometry()
                displacement = cog2 - cog1 
                distance = np.round(np.sqrt(displacement.dot(displacement)),3)
                atom_distances = self.get_molecule_atom_distances(i,j)
                if distance in com_distances:
                    continue
                elif ((distance > 5) & (np.min(atom_distances) > 5)):
                    continue
                else:
                    dimer = Molecule(atoms = mol1.atoms+mol2.atoms,
                                     bonds = mol1.bonds+mol2.bonds)
                    dimer.to_xyz(f'mol{i}_mol{j}_dimer')
                    obConversion = openbabel.OBConversion()
                    obConversion.SetInAndOutFormats("xyz", "mol")
                    mol = openbabel.OBMol()
                    obConversion.ReadFile(mol, f'mol{i}_mol{j}_dimer.xyz') 
                    mol.AddHydrogens()
                    obConversion.WriteFile(mol, f'mol{i}_mol{j}_dimer.mol')
                    shutil.copy(f'mol{i}_mol{j}_dimer.mol',f'../mol{i}_mol{j}_dimer.mol')
                    os.remove(f'mol{i}_mol{j}_dimer.xyz')
                    os.remove(f'mol{i}_mol{j}_dimer.mol')
                    file.write(f'mol{i}_mol{j}_dimer\n')
                com_distances.append(distance)  
        file.close()
        shutil.copy('names.txt','../names.txt') 
        os.remove('names.txt')
        os.chdir('..')
        os.rmdir('tempdir')
        
    def get_molecule_atom_distances(self,mol1_index,mol2_index):
        distances = []
        for atom1 in self.molecules[mol1_index].atoms:
            for atom2 in self.molecules[mol2_index].atoms:
                coords1 = atom1.coordinates
                coords2 = atom2.coordinates 
                disp = coords2 - coords1
                dist = np.round(np.sqrt(disp.dot(disp)),3)
                distances.append(dist)
        return distances

    def get_molecule_atom_vdw_distances(self,mol1_index,mol2_index):
        distances = []
        for atom1 in self.molecules[mol1_index].atoms:
            for atom2 in self.molecules[mol2_index].atoms:
                coords1 = atom1.coordinates
                coords2 = atom2.coordinates 
                disp = coords2 - coords1
                dist = np.round(np.sqrt(disp.dot(disp)),3)
                dist -= (atom1.vdw_radius + atom2.vdw_radius)
                distances.append(dist)
        return distances
        