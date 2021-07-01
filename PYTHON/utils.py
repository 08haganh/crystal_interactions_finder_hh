# File with example functions for generating data using the rest of the packages
# Basically a script file

from Atom import *
from Bond import *
from Crystal import *
from Geometry import *
from Interaction import *
from io import *
from Molecule import *
from Viz import *

def calc_intermolecular_atom_distances(crystal):
    '''
    Calculates all interatomic atom atom distances in a crystal structure
    calculates distances on batch between central molecules and a neighbour molecule, rather than a simple
    nested for loop
    calculates distances in batches between atom i in central molecule and atom (i - x) in neighbour
    returns a dataframe with all atom atom distances < 10A in the crystal structure
    '''
    # Calculate Atom Distances from central molecule
    central_molecule, central_idx = crystal.get_central_molecule(return_idx=True)
    central_atom_coords = np.array([atom.coordinates for atom in central_molecule.atoms]) # shape = (n_atoms,3)
    all_atom_coords = []
    for mol in crystal.molecules:
        all_atom_coords.append(np.array([atom.coordinates for atom in mol.atoms]))
    all_atom_coords = np.array(all_atom_coords) # shape = (n_mols,n_atoms,3) 
    dists = []
    mol1s = []
    mol2s = []
    for i, mol_coords in enumerate(all_atom_coords):
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
    # Put distances in order of atom indices
    in_atom_order = np.array([dist.flatten('F') for dist in dists]).reshape(-1)
    d1 = dists.shape[0]
    d2 = dists.shape[1]
    arange = np.arange(d2)
    atom1s = np.concatenate([[x]*d2 for x in range(d2)]*d1)
    atom2s = np.concatenate([np.roll(arange,-x) for x in range(d2)]*d1)
    #atom2s = np.concatenate([[x for x in range(d2)]*d2]*d1)
    # Turn Atom Distances to DataFrame
    data_dict= {'mol1s':mol1s,'mol2s':mol2s,'atom1s':atom1s,'atom2s':atom2s,'dists':in_atom_order}
    atom_dist_df = pd.DataFrame(data_dict)
    atom_dist_df = atom_dist_df[atom_dist_df.mol1s != atom_dist_df.mol2s]
    atom_dist_df = atom_dist_df.loc[atom_dist_df.dists <= 10]
    
    return atom_dist_df

def add_interactions(atom_dist_df,crystal):
    '''
    Add intermolecular interaction types to bond distances
    '''
    atom_dicts = []
    for idx in atom_dist_df.index:
        m1_idx = atom_dist_df.at[idx,'mol1s']
        m2_idx = atom_dist_df.at[idx,'mol2s']
        a1_idx = atom_dist_df.at[idx,'atom1s']
        a2_idx = atom_dist_df.at[idx,'atom2s']
        atom1 = crystal.molecules[m1_idx].atoms[a1_idx]
        atom2 = crystal.molecules[m2_idx].atoms[a2_idx]
        disp = atom2.coordinates - atom1.coordinates
        dist = np.sqrt(disp.dot(disp))
        atom_dict = {'mol1s':m1_idx,'mol2s':m2_idx,'atom1':a1_idx,'atom2':a2_idx,'dist':dist}
        interaction = Interaction(atom1,atom2)
        atom_dict.update(interaction.to_dict())
        atom_dicts.append(atom_dict)
    atom_df = pd.DataFrame(atom_dicts)
    
    return atom_df.set_index(['mol1s','mol2s'])

def calc_geometric_interactions(crystal):
    # Calculate Molecular Distances, Angles, Displacement
    all_mols = []
    for mol in crystal.molecules:
        all_mols.append(mol.centre_of_geometry())
    all_mols = np.array(all_mols)
    cog_disps = []
    cog_dists = []
    cog_mol1s = []
    cog_mol2s = []
    interplanar_angles = []
    unit_cell_disps = []
    planes = []
    # Rather than making new planes every loop, make the set of planes once
    for mol in crystal.molecules:
        planes.append(Plane(mol.get_backbone().atoms))
    # Loop through all pairs of molecules
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
    # Turn lists to arrays      
    unit_cell_disps = np.array(unit_cell_disps)        
    cog_dists = np.array(cog_dists)
    # Create Molecule Geometry to DataFrame
    data_dict= {'mol1s':cog_mol1s,'mol2s':cog_mol2s,
                'a':unit_cell_disps[:,0],'b':unit_cell_disps[:,1],'c':unit_cell_disps[:,2],
                'dists':cog_dists,'interplanar_angles':interplanar_angles}
    df_cogs = np.round(pd.DataFrame(data_dict).set_index(['mol1s','mol2s']),3)
    
    return df_cogs


def combine_topology_geometry(interaction_df,geometry_df):
     # Add to df_cogs
    hbond_bond = pd.DataFrame(interaction_df.groupby(interaction_df.index)['hydrogen_bond'].sum()).set_index(interaction_df.index.unique())
    pi_pi_bond = pd.DataFrame(interaction_df.groupby(interaction_df.index)['pi_pi_bond'].sum()).set_index(interaction_df.index.unique())
    halogen_bond = pd.DataFrame(interaction_df.groupby(interaction_df.index)['halogen_bond'].sum()).set_index(interaction_df.index.unique())
    ch_pi_bond = pd.DataFrame(interaction_df.groupby(interaction_df.index)['ch_pi_bond'].sum()).set_index(interaction_df.index.unique())
    hydrophobic_bond = pd.DataFrame(interaction_df.groupby(interaction_df.index)['hydrophobic_cc_bond'].sum()).set_index(interaction_df.index.unique())
    vdw_contact = pd.DataFrame(interaction_df.groupby(interaction_df.index)['vdw_contact'].sum()).set_index(interaction_df.index.unique())
    fin_df = pd.concat([geometry_df,vdw_contact,hbond_bond,pi_pi_bond,halogen_bond,ch_pi_bond,
                        hydrophobic_bond],axis=1)
    # Align interactions properly to remove double counting of indices
    double_counted = fin_df.loc[fin_df.index.get_level_values(0) > fin_df.index.get_level_values(1)]
    double_counted = double_counted[['vdw_contact','hydrogen_bond','pi_pi_bond','halogen_bond',
                                     'ch_pi_bond','hydrophobic_cc_bond']]
    fin_df.drop(double_counted.index,axis=0,inplace=True)
    arrays = [double_counted.index.get_level_values(1),double_counted.index.get_level_values(0)]
    tuples = list(zip(*arrays))
    double_counted.index = pd.MultiIndex.from_tuples(tuples, names=["mol1s", "mol2s"])
    fin_df.loc[double_counted.index,double_counted.columns] = double_counted

    return fin_df
