import numpy as np
from .Geometry import *

config = {'hydrogen_bond':{'dist_cutoff':4.1,'min_angle':150,'max_angle':180},
          'pi_pi_bond':{'dist_cutoff':5.5,'min_angle':0,'max_angle':30,'max_offset':4.0},
          'halogen_bond':{'dist_cutoff':4.0,'min_angle':0,'max_angle':30},
          'ch_pi_bond':{'dist_cutoff':4.0,'min_angle':0,'max_angle':50},
          'hydrophobic_cc_bond':{'dist_cutoff':4.0}}

def assign_atom_interaction_dict(atom):
    # Define interaction types
    interaction_dict = {'hydrogen_bond':{'acceptor':False,
                                     'donor':False},
                    'pi_pi_bond':{'acceptor':False,
                                  'donor':False},
                    'halogen_bond':{'acceptor':False,
                                    'donor':False},
                    'ch_pi_bond':{'acceptor':False,
                                  'donor':False},
                    'hydrophobic_cc_bond':{'acceptor':False,
                                           'donor':False}}

    # Code to assign whether atom is X Bond acceptor and/or donor
    # Hydrogen Bond Acceptor
    if atom.symbol in ['O','N','F','S']:
        interaction_dict['hydrogen_bond']['acceptor'] = True 
    # Hydrogen Bond Donor
    if atom.symbol == 'H':
        all_bonded_atoms = []
        for bond in atom.bonds:
            all_bonded_atoms.append(bond.atom1.symbol)
            all_bonded_atoms.append(bond.atom2.symbol)
        if  np.sum(np.isin(np.array(['O','N','F']),np.array(all_bonded_atoms))) > 0:
            interaction_dict['hydrogen_bond']['donor'] = True 
    # Pi bond acceptor and donor
    if ((atom.symbol == 'ring') & (atom.type == 'aromatic')):
        interaction_dict['pi_pi_bond']['acceptor'] = True
        interaction_dict['pi_pi_bond']['donor'] = True
    # Halogen Bond Acceptor
    if atom.symbol in ['O','N','S','Se','P']:
        interaction_dict['halogen_bond']['acceptor'] = True
    # Halogen Bond Donor
    if atom.symbol in ['F','CL','Cl','BR','Br','I']:
        interaction_dict['halogen_bond']['donor'] = True
    # CH pi Bond Acceptor
    if ((atom.symbol == 'ring') & (atom.type == 'aromatic')):
        interaction_dict['ch_pi_bond']['acceptor'] = True
    # CH pi Bond Donor
    if atom.symbol == 'H':
        all_bonded_atoms = []
        for bond in atom.bonds:
            all_bonded_atoms.append(bond.atom1.symbol)
            all_bonded_atoms.append(bond.atom2.symbol)
        if  np.sum(np.isin(np.array(['C']),np.array(all_bonded_atoms))) > 0:
            interaction_dict['ch_pi_bond']['donor'] = True 
    # Hydrophobic C-C bond Acceptor and Donor
    if atom.symbol == 'C':
        all_bonded_atoms = []
        for bond in atom.bonds:
            all_bonded_atoms.append(bond.atom1.symbol)
            all_bonded_atoms.append(bond.atom2.symbol)
        if  np.sum(np.isin(np.array(all_bonded_atoms),np.array(['C','H']),invert=True)) == 0:
            interaction_dict['hydrophobic_cc_bond']['donor'] = True
            interaction_dict['hydrophobic_cc_bond']['acceptor'] = True 

    return interaction_dict

class Interaction():
    def __init__(self,atom1,atom2,basis):
        self.atom1 = atom1 
        self.atom2 = atom2 
        self.atoms = [atom1, atom2]
        self.basis = basis
        self.vdw_sum = self.atom1.vdw_radius + self.atom2.vdw_radius
        self.vdw_contact = self.length() < self.vdw_sum
        self.vdw_distance = self.length() - self.vdw_sum
        self.assign_interaction()

    def assign_interaction(self):
        self.types = {'hydrogen_bond':0,'pi_pi_bond':0,'halogen_bond':0,'ch_pi_bond':0,'hydrophobic_cc_bond':0,
                      'bond_angle':np.nan,'theta1':np.nan,'theta2':np.nan} 
        distance = self.length()
        # Assign whether hydrogen bond
        hydrogen_donor_acceptor = (((self.atom1.interaction_dict['hydrogen_bond']['acceptor']) & 
                                    (self.atom2.interaction_dict['hydrogen_bond']['donor'])) | 
                                    ((self.atom2.interaction_dict['hydrogen_bond']['acceptor']) & 
                                    (self.atom1.interaction_dict['hydrogen_bond']['donor'])))
        hydrogen_within_distance = distance < config['hydrogen_bond']['dist_cutoff']
        if hydrogen_donor_acceptor & hydrogen_within_distance:
            # Assign hydrogen bond angle
            # Angle between X-H--D 
            if self.atom1.interaction_dict['hydrogen_bond']['donor']:
                assert len(self.atom1.neighbours) == 1
                neighbour = self.atom1.neighbours[0]
                hbond_angle = bond_angle(neighbour,self.atom1,self.atom2)
                if ((hbond_angle > config['hydrogen_bond']['min_angle']) & (hbond_angle < config['hydrogen_bond']['max_angle'])):
                    self.types['hydrogen_bond'] = 1
                    self.types['bond_angle'] = hbond_angle
            else:
                assert len(self.atom2.neighbours) == 1
                neighbour = self.atom2.neighbours[0]
                hbond_angle = bond_angle(neighbour,self.atom2,self.atom1)
                if ((hbond_angle > config['hydrogen_bond']['min_angle']) & (hbond_angle < config['hydrogen_bond']['max_angle'])):
                    self.types['hydrogen_bond'] = 1
                    self.types['bond_angle'] = hbond_angle
        # Assign whether pi-pi bond
        pi_donor_acceptor = (((self.atom1.interaction_dict['pi_pi_bond']['acceptor']) & 
                                    (self.atom2.interaction_dict['pi_pi_bond']['donor'])) | 
                                    ((self.atom2.interaction_dict['pi_pi_bond']['acceptor']) & 
                                    (self.atom1.interaction_dict['pi_pi_bond']['donor'])))
        pi_within_distance = distance < config['pi_pi_bond']['dist_cutoff']
        if pi_donor_acceptor & pi_within_distance:
            # Calculate bond angle
            # Angle between pi-pi bond and plane of ring1
            pi_plane1 = Plane(self.atom1.ring_atoms)
            pi_plane2 = Plane(self.atom2.ring_atoms)
            pi_bond_angle = pi_plane1.plane_angle(pi_plane2)
            # Calcaulating offset
            disp = self.atom2.coordinates - self.atom1.coordinates
            mol_disp = calc_lstsq_displacement(disp, self.basis)
            if np.sqrt(np.dot(mol_disp[0],mol_disp[2])) < config['pi_pi_bond']['max_offset']:
                if pi_bond_angle > 90:
                    pi_bond_angle = 180 - pi_bond_angle
                pi_within_angle = ((pi_bond_angle > config['pi_pi_bond']['min_angle']) & (pi_bond_angle < config['pi_pi_bond']['max_angle']))
                if pi_within_angle:
                    self.types['pi_pi_bond'] = 1
                    self.types['bond_angle'] = pi_bond_angle
        # Assign whether halogen bond
        halogen_donor_acceptor = (((self.atom1.interaction_dict['halogen_bond']['acceptor']) & 
                                    (self.atom2.interaction_dict['halogen_bond']['donor'])) | 
                                    ((self.atom2.interaction_dict['halogen_bond']['acceptor']) & 
                                    (self.atom1.interaction_dict['halogen_bond']['donor'])))
        halogen_within_distance = distance < config['halogen_bond']['dist_cutoff']
        if halogen_donor_acceptor & halogen_within_distance:
            if len(self.atom1.neighbours) == 1:
                neighbour = self.atom1.neighbours[0]
                theta1 = bond_angle(neighbour,self.atom1,self.atom2)
                self.types['theta1'] = theta1
                if np.isclose(theta1, 90, atol=30) | np.isclose(theta1, 180, atol=30):
                    self.types['halogen_bond'] = 1
            if len(self.atom2.neighbours) == 1:
                neighbour = self.atom2.neighbours[0]
                theta2 = bond_angle(neighbour,self.atom2,self.atom1)
                self.types['theta2'] = theta2
                if np.isclose(theta2, 90, atol=30) | np.isclose(theta2, 180, atol=30):
                    self.types['halogen_bond'] = 1

        # Assign whether CH-pi bond
        ch_pi_donor_acceptor = (((self.atom1.interaction_dict['ch_pi_bond']['acceptor']) & 
                                    (self.atom2.interaction_dict['ch_pi_bond']['donor'])) | 
                                    ((self.atom2.interaction_dict['ch_pi_bond']['acceptor']) & 
                                    (self.atom1.interaction_dict['ch_pi_bond']['donor'])))
        ch_pi_within_distance = distance < config['ch_pi_bond']['dist_cutoff']
        if ch_pi_donor_acceptor & ch_pi_within_distance:
            # Calculate Bond Angle
            # angle between H-pi bond vector and plane of the ring
            if self.atom1.interaction_dict['ch_pi_bond']['acceptor']:
                pi_plane = Plane(self.atom1.ring_atoms)
                pi_norm = np.array([pi_plane.a,pi_plane.b,pi_plane.c])
                disp = self.atom2.coordinates - self.atom1.coordinates
                pi_bond_angle = np.degrees(np.arccos(disp.dot(pi_norm)/(np.sqrt(disp.dot(disp))*np.sqrt(pi_norm.dot(pi_norm)))))
                if pi_bond_angle > 90:
                    pi_bond_angle = 180 - pi_bond_angle
                pi_within_angle = ((pi_bond_angle > config['ch_pi_bond']['min_angle']) & (pi_bond_angle < config['ch_pi_bond']['max_angle']))
                if pi_within_angle:
                    self.types['ch_pi_bond'] = 1
                    self.types['bond_angle'] = pi_bond_angle
            else:
                pi_plane = Plane(self.atom2.ring_atoms)
                pi_norm = np.array([pi_plane.a,pi_plane.b,pi_plane.c])
                disp = self.atom2.coordinates - self.atom1.coordinates
                pi_bond_angle = np.degrees(np.arccos(disp.dot(pi_norm)/(np.sqrt(disp.dot(disp))*np.sqrt(pi_norm.dot(pi_norm)))))
                if pi_bond_angle > 90:
                    pi_bond_angle = 180 - pi_bond_angle
                pi_within_angle = ((pi_bond_angle > config['ch_pi_bond']['min_angle']) & (pi_bond_angle < config['ch_pi_bond']['max_angle']))
                if pi_within_angle:
                    self.types['ch_pi_bond'] = 1
                    self.types['bond_angle'] = pi_bond_angle
        # Hydrophobic Interactions
        hydrophobic_donor_acceptor = (((self.atom1.interaction_dict['hydrophobic_cc_bond']['acceptor']) & 
                                    (self.atom2.interaction_dict['hydrophobic_cc_bond']['donor'])) | 
                                    ((self.atom2.interaction_dict['hydrophobic_cc_bond']['acceptor']) & 
                                    (self.atom1.interaction_dict['hydrophobic_cc_bond']['donor'])))
        hydrophobic_within_distance = distance < config['hydrophobic_cc_bond']['dist_cutoff']
        if hydrophobic_donor_acceptor & hydrophobic_within_distance:
            self.types['hydrophobic_cc_bond'] = 1


    def to_dict(self):
        info_dict = {'atom1_symbol':self.atom1.symbol,'atom2_symbol':self.atom2.symbol,'dist':self.length(),'vdw_sum':self.vdw_sum,
                     'vdw_contact':self.vdw_contact,'vdw_distance':self.vdw_distance}
        info_dict.update(self.types)
        return info_dict


    def length(self):
        disp = self.atom2.coordinates - self.atom1.coordinates 
        return np.sqrt(disp.dot(disp))




    