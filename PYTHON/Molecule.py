'''
File to store Molecule class and classes that inherit it such as Acene, Helicene, etc.
Molecule Class
Attributes
- atoms; list; list of atom objects
- bonds; list; list of bond objects
- ring_atoms; nested list; list of lists of atom objects in rings in molecule
- ring_bonds; nested_list; list of lists of bond objects in rings in molecule
- rings; list; list of Molecule objects of the all rings in the molecule
- pseudo_atoms; list; list of 'pseudo atoms' in the molecule e.g. ring centroids; 
note appropriate methods must be called to populate
Methods
- assign_atom_bonds(); return None; assigns bonds attribute to atom objects in the molecule
- assign_atom_neighbours(); return None; assigns atom_neighbours attribute to atom objects in the Molecule
- assign_rings(); return None; finds rings and assigns them to rings attribute
- add_rings_as_pseudo(); return None; adds ring centroids as 'ring' atom objects
- centre_of_geometry(); return np.array; returns the centre of geometry of the molecule
- to_nx_graph(); return Networkx graph object; converts the molecule to an 
nx graph object with atoms as nodes and bonds as edges
- get_fused_ring_systems(); return list; returns a list of fused ring systems in the molecule e.g. acene backbone
- to_xyz(filename); return None; saves the moleucule as an xyz file 
- get_components(); return list; returns a list of non-bonded molecules in molecule object
'''
import numpy as np
import networkx as nx
from .Bond import Bond
from .Atom import Atom, RingCentroid
from .Interaction import *
from .Geometry import *
import pandas as pd

class Molecule():
    def __init__(self,atoms=[],bonds=[]):
        self.atoms = atoms
        self.bonds = bonds
        self.ring_atoms = []
        self.ring_bonds = []
        self.rings = []
        
    def assign_atom_bonds(self):
        for bond in self.bonds:
            for atom in bond.atoms:
                atom.bonds.append(bond)

    def get_ring_atoms(self):
        if len(self.ring_atoms) > 0:
            return self.ring_atoms
        else: 
            self.assign_rings()
            return self.ring_atoms

    def assign_atoms_interaction_dict(self):
        for atom in self.atoms:
            atom.interaction_dict = assign_atom_interaction_dict(atom)

    def assign_rings(self):
        self.ring_atoms = nx.cycle_basis(self.to_nx_graph())
        self.ring_bonds = []
        self.rings = []
        for ring in self.ring_atoms:
            ring_bonds = []
            for bond in self.bonds:
                if np.sum(np.isin(bond.atoms,ring)) == 2:
                    ring_bonds.append(bond)
                else:
                    continue 
            self.ring_bonds.append(ring_bonds)
        for ring_atoms, ring_bonds in zip(self.ring_atoms,self.ring_bonds):
            ring = Molecule(ring_atoms,ring_bonds)
            self.rings.append(ring)

    def assign_atom_neighbours(self):
        for atom in self.atoms:
            for bond in atom.bonds:
                for neighbour in bond.atoms:
                    atom.neighbours.append(neighbour) if neighbour != atom else 0
    
    def add_rings_as_atoms(self):
        # Need to update this function to differentiate between aromatic and aliphatic rings
        # Atm no need as we are working with fully conjugated acene dataset
        if len(self.ring_atoms) == 0:
            self.assign_rings()
        ring_count = 1
        for atoms, bonds in zip(self.ring_atoms,self.ring_bonds):
            temp_mol = Molecule(atoms,bonds)
            cog = temp_mol.centre_of_geometry()
            bond_lengths = [bond.length() for bond in bonds]
            if np.mean(bond_lengths) < 1.45:
                pseudo_atom = RingCentroid(ring_atoms=atoms,atom_symbol='ring',atom_type='aromatic',coordinates=cog,label=f'ring{ring_count}')
            else:
                pseudo_atom = RingCentroid(ring_atoms=atoms,atom_symbol='ring',atom_type='aliphatic',coordinates=cog,label=f'ring{ring_count}') 
            self.atoms.append(pseudo_atom)

    def get_fused_ring_systems(self):
        all_fused_ring_systems = Molecule([atom for ring in self.ring_atoms for atom in ring],
                                          [bond for ring in self.ring_bonds for bond in ring])
        return all_fused_ring_systems.get_components()
        
    def to_nx_graph(self):
        G = nx.Graph()
        for atom in self.atoms:
            G.add_node(atom)
        for bond in self.bonds:
            G.add_edge(bond.atom1,bond.atom2,order=bond.order)
        return G

    def centre_of_geometry(self,ignore_rings=True):
        atom_positions = []
        for atom in self.atoms:
            if ignore_rings:
                if atom.symbol == 'ring':
                    continue
            atom_positions.append(atom.coordinates)
        atom_positions = np.array(atom_positions)
        atom_positions = atom_positions.reshape(atom_positions.shape[0],3)
        return np.mean(atom_positions,axis=0)

    def test_planarity(self):
        mol_plane = Plane(self.atoms)
        devs = [mol_plane.point_distance(atom) for atom in self.atoms]
        if np.mean(devs) > 1:
            return False
        else:
            return True

    def to_mol2(self):
        pass

    def get_components(self):
        g = nx.Graph()
        for atom in self.atoms:
            g.add_node(atom)
        for bond in self.bonds:
            g.add_edge(bond.atom1,bond.atom2,order=bond.order)
        subgraphs = [g.subgraph(c) for c in nx.connected_components(g)] 
        molecules = []
        for graph in subgraphs:
            bonds = []
            for edge in graph.edges:
                bonds.append(Bond(edge[0],edge[1],g[edge[0]][edge[1]]['order']))
            mol = Molecule(list(graph.nodes),bonds)
            molecules.append(mol)
        return molecules

    def to_xyz(self,filename):
        atom_symbols = np.array([atom.symbol for atom in self.atoms])
        unique_atoms = np.unique(atom_symbols)
        atom_count_dict = {}
        for unique_atom in unique_atoms:
            atom_count_dict[unique_atom] = np.sum(np.isin(atom_symbols,unique_atom))
        with open(f'{filename}.xyz','w') as file:
            file.write(f'{len(self.atoms)}\n')
            for key in atom_count_dict.keys():
                file.write(f'{key}{atom_count_dict[key]} ')
            file.write('\n')
            for atom in self.atoms:
                coords = atom.coordinates
                file.write(f'{atom.symbol} {coords[0]} {coords[1]} {coords[2]}\n')
        return None

class Acene(Molecule):
    def __init__(self,molecule):
        self.atoms = [atom for atom in molecule.atoms]
        self.bonds = [bond for bond in molecule.bonds]
        self.ring_atoms = [ring_atom for ring_atom in molecule.ring_atoms]
        self.ring_bonds = [ring_bond for ring_bond in molecule.ring_bonds]
        self.rings = [ring for ring in molecule.rings]

    def get_backbone(self):
        fused_ring_systems = self.get_fused_ring_systems()
        biggest_frs_size = 0
        for frs in fused_ring_systems:
            n_rings = len(frs.get_ring_atoms())
            if n_rings == np.max([biggest_frs_size,n_rings]):
                biggest_frs = frs
                biggest_frs_size = n_rings
        return Acene(biggest_frs)

    def get_peripheries(self):
        pass 

    def get_substitution_positions(self):
        pass 

    def get_backbone_heteroatoms(self):
        pass

    def get_vectors(self, return_xmag = False):
        # returns vectors corresponding to directions in terms of acene molecule, all scaled to 1A
        # x axis is the vector between ring centroids in the backbone
        # y axis orthogonal to the plane of the acene molecule
        # z axis is the vector that is the width of the acene
        # These three vectors makes an orthogonal basis set for R3
        # x and y vectors angle < 90.3, so approximately right angles
        bb = self.get_backbone()
        bb_cog = bb.centre_of_geometry()
        plane = Plane(bb.atoms)
        bb.assign_rings()
        ring_centroids = []
        ring_centroid_distances = []
        for ring in bb.rings:
            ring_cog = ring.centre_of_geometry()
            ring_disp = ring_cog - bb_cog
            ring_dist = np.sqrt(ring_disp.dot(ring_disp))
            ring_centroids.append(ring_cog)
            ring_centroid_distances.append(ring_dist)
        df = pd.DataFrame([ring_centroids,ring_centroid_distances]).T
        df.columns= ['centroid','cog_distance'] 
        df.sort_values('cog_distance',ascending=True,inplace=True)
        terminal_centroids = df.centroid[-2:].values
        o_trc = terminal_centroids[-2] - terminal_centroids[-1]
        normal = np.array([plane.a,plane.b,plane.c])
        theta = np.arccos((o_trc.dot(normal))/(np.sqrt(o_trc.dot(o_trc))*np.sqrt(normal.dot(normal))))
        if theta == np.pi/2:
            x_vect = o_trc
        else:
            x_vect = o_trc*np.sin(theta)
        y_vect = normal
        x_mag = np.sqrt(x_vect.dot(x_vect))
        y_mag = np.sqrt(y_vect.dot(y_vect))
        z_vect = np.cross(x_vect,y_vect)
        z_mag = np.sqrt(z_vect.dot(z_vect))
        for i, x in enumerate(x_vect):
            x_vect[i] = x * 1/x_mag
        for i, y in enumerate(y_vect):
            y_vect[i] = y * 1/y_mag
        for i, z in enumerate(z_vect):
            z_vect[i] = z * 1/z_mag 

        if return_xmag:
            return np.vstack([x_vect, y_vect, z_vect]), x_mag
        else:
            return np.vstack([x_vect, y_vect, z_vect])

    def label_backbone(self):
        pass