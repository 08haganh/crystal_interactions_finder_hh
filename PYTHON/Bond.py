'''
File to store the Bond class
Attributes
- atom1; atom object of atom1 of the bond
- atom2; atom object of atom2 of the bond
- order; string; order of the bond
- atoms; np.array; array of atom objects of the atoms involved in the bond
Methods
- length(); return float; returns the length of the bond
'''
import numpy as np
class Bond():
    def __init__(self,atom1,atom2,order):
        self.atom1 = atom1
        self.atom2 = atom2
        self.order = order
        self.atoms = np.array([atom1,atom2])
    def length(self):
        displacement = self.atom2.coordinates - self.atom1.coordinates
        return np.sqrt(displacement.dot(displacement))