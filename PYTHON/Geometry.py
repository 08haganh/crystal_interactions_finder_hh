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

# https://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
# Python implementation of the MATLAB function MinVolEllipse, based on the Khachiyan algorithm
# for both 
# A is a matrix containing the information regarding the shape of the ellipsoid 
# to get radii from A you have to do SVD on it, giving U Q and V
# 1 / sqrt(Q) gives the radii of the ellipsoid
# problems arise for planar motifs. add two extra points at centroid of +/- 0.00001*plane_normal to overcome
def mvee(atoms, tol = 0.00001):
    """
    Find the minimum volume ellipse around a set of atom objects.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1
    [U Q V] = svd(A); 
    where r = 1/sqrt(Q)
    V is rotation matrix
    U is ??? 
    """
    points_asarray = np.array([atom.coordinates for atom in atoms])
    points = np.asmatrix(points_asarray)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    try:
        while err > tol:
            # assert u.sum() == 1 # invariant
            X = Q * np.diag(u) * Q.T
            M = np.diag(Q.T * la.inv(X) * Q)
            jdx = np.argmax(M)
            step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
            new_u = (1-step_size)*u
            new_u[jdx] += step_size
            err = la.norm(new_u-u)
            u = new_u
        c = u*points
        A = la.inv(points.T*np.diag(u)*points - c.T*c)/d    
    except: # For singular matrix errors i.e. motif is ellipse rather than ellipsoid
        centroid = np.average(points_asarray,axis=0)
        plane = Plane(atoms)
        normal = np.array([plane.a,plane.b,plane.c])
        norm_mag = np.sqrt(np.dot(normal,normal))
        for i, norm in enumerate(normal):
            normal[i] = norm * 1 / norm_mag
        centroid = np.average(points,axis=0).reshape(-1,3)
        p1 = centroid + normal*0.00001
        p2 = centroid - normal*0.00001
        points_asarray = np.concatenate([points_asarray,p1,p2],axis=0)
        points = np.asmatrix(points_asarray)
        N, d = points.shape
        Q = np.column_stack((points, np.ones(N))).T
        err = tol+1.0
        u = np.ones(N)/N
        while err > tol:
            # assert u.sum() == 1 # invariant
            X = Q * np.diag(u) * Q.T
            M = np.diag(Q.T * la.inv(X) * Q)
            jdx = np.argmax(M)
            step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
            new_u = (1-step_size)*u
            new_u[jdx] += step_size
            err = la.norm(new_u-u)
            u = new_u
        c = u*points
        A = la.inv(points.T*np.diag(u)*points - c.T*c)/d   
        
    return np.asarray(A), np.squeeze(np.asarray(c))

def ellipse(rx,ry,rz):
    u, v = np.mgrid[0:2*np.pi:20j, -np.pi/2:np.pi/2:10j]
    x = rx*np.cos(u)*np.cos(v)
    y = ry*np.sin(u)*np.cos(v)
    z = rz*np.sin(v)
    return x,y,z