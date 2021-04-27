''' 
File to store Atom class
Attributes
- symbol; string; 'C', 'N'
- type; string; C.ar
- coordinates; np.array; array([1,0,0])
- vdw_radius; float
- mass; float
- label; string
- bonds; list; list of bond objects
- neighbour; list; list of atom objects
- interaction_dict; dictionary; dictionary of interaction types the atom can partake in
Methods
- translate(new_coordinates); return None; moves atom to new position
- rotate(about,angle); return None; rotates atom about point (x,y,z) by angle
- remove(); return None; deletes atom
- assign_label(label); return None; relabels the atom to label
- get_parent_molecule_index(); return int; return the index of the parent molecule in the crystal
'''
import numpy as np
from .Geometry import *

class Atom():
    def __init__(self,atom_symbol,atom_type,coordinates,label=''):
        self.symbol = atom_symbol
        self.type = atom_type
        self.coordinates = np.array(coordinates)  
        self.label = label
        self.bonds = []
        self.neighbours = []
        self.interaction_dict = {}
        try:
            self.mass = atomic_mass[self.symbol]
        except:
            self.mass = 0
        try:
            self.vdw_radius = vdw_radii[self.symbol]
        except:
            self.vdw_radius = 0

    def __str__(self):
        return '{self.label}'.format(self=self)

    def translate(self,new_position):
        self.coordinates = new_position

    def rotate(self,about,angle):
        pass 

    def remove(self):
        del self 
    def assign_label(self,label):
        self.label = label

    def get_parent_molecule_index(self,crystal):
        for i, molecule in enumerate(crystal.molecules):
            if self in molecule.atoms:
                return i 
        raise 'Atom not found in crystal'

class RingCentroid(Atom):
    def __init__(self,ring_atoms,atom_symbol,atom_type,coordinates,label=''):
        self.symbol = atom_symbol
        self.type = atom_type
        self.coordinates = np.array(coordinates) 
        self.ring_atoms = [atom for atom in ring_atoms] 
        self.label = label
        self.bonds = []
        self.neighbours = []
        self.interaction_dict = {}
        self.has_vectors = False
        try:
            self.mass = atomic_mass[self.symbol]
        except:
            self.mass = 0
        try:
            self.vdw_radius = vdw_radii[self.symbol]
        except:
            self.vdw_radius = 0

    def get_vectors(self):
        self.plane = Plane(self.ring_atoms)
        y_vect = np.array([self.plane.a,self.plane.b,self.plane.c])
        x_vect = self.ring_atoms[0].coordinates - self.coordinates
        z_vect = np.cross(x_vect,y_vect)
        assert np.round(vector_angle(x_vect,y_vect),0) == 90
        assert np.round(vector_angle(x_vect,z_vect),0) == 90
        assert np.round(vector_angle(z_vect,y_vect),0) == 90
        x_mag = np.sqrt(x_vect.dot(x_vect))
        y_mag = np.sqrt(y_vect.dot(y_vect))
        z_mag = np.sqrt(z_vect.dot(z_vect))
        for i, x in enumerate(x_vect):
            x_vect[i] = x * 1/x_mag
        for i, y in enumerate(y_vect):
            y_vect[i] = y * 1/y_mag
        for i, z in enumerate(z_vect):
            z_vect[i] = z * 1/z_mag 

        self.vectors = np.vstack([x_vect, y_vect, z_vect])
        self.has_vectors = True    



vdw_radii = {
              'Al': 2, 'Sb': 2, 'Ar': 1.88, 'As': 1.85, 'Ba': 2,
              'Be': 2, 'Bi': 2, 'B': 2, 'Br': 1.85, 'Cd': 1.58,
              'Cs': 2, 'Ca': 2, 'C': 1.7, 'Ce': 2, 'Cl': 1.75,
              'Cr': 2, 'Co': 2, 'Cu': 1.4, 'Dy': 2, 'Er': 2,
              'Eu': 2, 'F':  1.47, 'Gd': 2, 'Ga': 1.87, 'Ge': 2,
              'Au': 1.66, 'Hf': 2, 'He': 1.4, 'Ho': 2, 'H': 1.09,
              'In': 1.93, 'I': 1.98, 'Ir': 2, 'Fe': 2, 'Kr': 2.02,
              'La': 2, 'Pb': 2.02, 'Li': 1.82, 'Lu': 2, 'Mg': 1.73,
              'Mn': 2, 'Hg': 1.55, 'Mo': 2, 'Nd': 2, 'Ne': 1.54,
              'Ni': 1.63, 'Nb': 2, 'N':  1.55, 'Npl':  1.55, 'Os': 2,
              'O': 1.52,
              'Pd': 1.63, 'P': 1.8, 'Pt': 1.72, 'K': 2.75, 'Pr': 2,
              'Pa': 2, 'Re': 2, 'Rh': 2, 'Rb': 2, 'Ru': 2, 'Sm': 2,
              'Sc': 2, 'Se': 1.9, 'Si': 2.1, 'Ag': 1.72, 'Na': 2.27,
              'Sr': 2, 'S': 1.8, 'Ta': 2, 'Te': 2.06, 'Tb': 2,
              'Tl': 1.96, 'Th': 2, 'Tm': 2, 'Sn': 2.17, 'Ti': 2,
              'W': 2, 'U':  1.86, 'V':  2, 'Xe': 2.16, 'Yb': 2,
              'Y': 2, 'Zn': 1.29, 'Zr': 2, 'X':  1.0, 'D':  1.0,
              'O2': 1.52,'ring':0,
              'AL': 2, 'SB': 2, 'AR': 1.88, 'AS': 1.85, 'BA': 2,
              'BE': 2, 'BI': 2, 'B': 2, 'BR': 1.85, 'CD': 1.58,
              'CS': 2, 'CA': 2, 'C': 1.7, 'CE': 2, 'CL': 1.75,
              'CR': 2, 'CO': 2, 'CU': 1.4, 'DY': 2, 'ER': 2,
              'EU': 2, 'F':  1.47, 'GD': 2, 'GA': 1.87, 'GE': 2,
              'AU': 1.66, 'HF': 2, 'HE': 1.4, 'HL': 2, 'H': 1.09,
              'IN': 1.93, 'I': 1.98, 'IR': 2, 'FE': 2, 'KR': 2.02,
              'LA': 2, 'PB': 2.02, 'LI': 1.82, 'LU': 2, 'MG': 1.73,
              'MN': 2, 'HG': 1.55, 'MO': 2, 'ND': 2, 'NE': 1.54,
              'NI': 1.63, 'NB': 2, 'N':  1.55, 'NPL':  1.55, 'OS': 2,
              'O': 1.52,
              'PD': 1.63, 'P': 1.8, 'PT': 1.72, 'K': 2.75, 'PR': 2,
              'PA': 2, 'RE': 2, 'RH': 2, 'RB': 2, 'RU': 2, 'SM': 2,
              'SC': 2, 'SE': 1.9, 'SI': 2.1, 'AG': 1.72, 'NA': 2.27,
              'SR': 2, 'S': 1.8, 'TA': 2, 'TE': 2.06, 'TB': 2,
              'TL': 1.96, 'TH': 2, 'TM': 2, 'SN': 2.17, 'TI': 2,
              'W': 2, 'U':  1.86, 'V':  2, 'XE': 2.16, 'YB': 2,
              'Y': 2, 'ZN': 1.29, 'ZR': 2, 'X':  1.0, 'D':  1.0,
              'O2': 1.52,'ring':0
                 }

atomic_mass = {
                'H':1.0079, 'He':4.0026, 'Li':6.941, 'Be':9.0122, 'B':10.811,
                'C':12.0107, 'N': 14.0067, 'O':15.9994, 'F':18.9984, 'Ne':20.1797,
                'Na':22.9897, 'Mg':24.305, 'Al':26.9815, 'Si':28.0855, 'P':30.9738,
                'S':32.065, 'Cl':35.453, 'K':39.0983, 'Ar':39.948, 'Ca':40.078,
                'Sc':44.9559, 'Ti':47.867, 'V':50.9415, 'Cr':51.9961, 'Mn':54.938,
                'Fe':55.845, 'Ni':58.6934, 'Co':58.9332, 'Cu':63.546, 'Zn':65.39
                }