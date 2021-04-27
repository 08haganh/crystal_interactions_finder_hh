import math
import numpy as np

class Plane():
    def __init__(self,atoms):
        # Stores a plane equation in the format
        # ax + bx + cz + d = 0
        self.atoms = atoms
        xs = [atom.coordinates[0] for atom in atoms]
        ys = [atom.coordinates[1] for atom in atoms]
        zs = [atom.coordinates[2] for atom in atoms]
        # do fit
        tmp_A = []
        tmp_b = []
        for i in range(len(xs)):
            tmp_A.append([xs[i], ys[i], 1])
            tmp_b.append(zs[i])
        b = np.matrix(tmp_b).T
        A = np.matrix(tmp_A)
        fit =  (A.T * A).I * A.T * b
        self.errors = b - A * fit
        fit = np.array(fit).reshape(3)
        self.a, self.b, self.d = fit[0], fit[1], fit[2]

        # fit is currently in the form
        # ax + by + d = cz
        # c = -(a*x[0] + b*y[0] + d) / z[0]
        self.c = - ((self.a*xs[0] + self.b*ys[0] + self.d) / zs[0])
        
    def plane_angle(self, plane):
        a1,b1,c1 = self.a,self.b, self.c
        a2,b2,c2 = plane.a,plane.b, plane.c
            
        d = ( a1 * a2 + b1 * b2 + c1 * c2 )
        e1 = np.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
        e2 = np.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
        d = d / (e1 * e2)
        A = np.degrees(np.arccos(d))
        if A > 90:
            A = 180 - A
        return A

    def point_distance(self,atom): 
        x1, y1, z1 = atom.coordinates[0], atom.coordinates[1], atom.coordinates[2]
        d = np.abs((self.a * x1 + self.b * y1 + self.c * z1 + self.d)) 
        e = (np.sqrt(self.a * self.a + self.b * self.b + self.c * self.c))
        return d/e

    def test_planarity(self,atoms = None):
        if atoms == None:
            devs = [self.point_distance(atom) for atom in self.atoms]
            if len(np.where(np.array(devs)>2)[0]) >= 1:
                return False
            else:
                return True
        else:
            devs = [self.point_distance(atom) for atom in atoms]
            if len(np.where(np.array(devs)>2)[0]) >= 1:
                return False
            else:
                return True

def bond_angle(atom1,atom2,atom3):
    a = atom1.coordinates
    b = atom2.coordinates
    c = atom3.coordinates

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def torsional_angle(atom1,atom2,atom3,atom4):
    # returns interplanar angle between planes defined by atom1, atom2, atom3, and atom2, atom3, atom4
    pass
def vector(atom1,atom2, as_angstrom=False):
    # returns the vector defined by the position between two atoms
    pass
def calc_lstsq_displacement(disp,vectors):
    A = vectors.T
    xs = []
    x, _, _, _ = np.linalg.lstsq(A,disp,rcond=-1)
    xs.append(x)
    return np.array(xs[0])
def vector_angle(v1,v2):
    theta = np.arccos((v1.dot(v2))/(np.sqrt(v1.dot(v1))*np.sqrt(v2.dot(v2))))
    return np.degrees(theta)
def vector_plane_angle(vector, plane):
    # returns the angle made between a vector and a plane
    pass