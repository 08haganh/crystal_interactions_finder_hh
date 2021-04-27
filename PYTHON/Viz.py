import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def network_plot_3D(G, angle, save=False):
    colours = ['black',
               'firebrick',
               'sandybrown',
               'orange',
               'gold',
               'lawngreen',
               'forestgreen',
               'mediumturquoise',
               'dodgerblue',
               'lightslategray',
               'navy',
               'blueviolet',
               'fuchsia',
               'pink']
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    # Get number of nodes
    n = G.number_of_nodes()
    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree(key) for key in pos.keys()])
    # Define color range proportional to number of edges adjacent to a single node
    colors = [plt.cm.plasma(G.degree(key)/edge_max) for key in pos.keys()] 
    # Get edge type indices for each edge
    unique_edges = []
    edge_info = []
    for x in G.edges.data('info'):
        unique_edges.append(x[2]) if x[2] not in edge_info else 0 
        edge_info.append(x[2])
    edge_types = [unique_edges.index(edge[2]) for edge in G.edges.data('info')]
    # 3D network plot
    with plt.style.context(('ggplot')):
        
        fig = plt.figure(figsize=(12,8),dpi=100)
        ax = Axes3D(fig)
        
        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        counter = 0
        
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]
            
            # Scatter plot
            ax.scatter(xi, yi, zi, color='black', s=20, edgecolors='k', alpha=0.7)
            counter += 1
        
        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        edges_encountered = []
        for i,j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))
        
        # Plot the connecting lines
            if edge_types[i] not in edges_encountered:
                ax.plot(x, y, z, c=colours[edge_types[i]],alpha=0.5, label=edge_info[i])
                edges_encountered.append(edge_types[i])
            else:
                ax.plot(x, y, z, c=colours[edge_types[i]], alpha=0.5)
        plt.legend()
    # Label Diagram
    ax.set_xlabel('a (Angstrom)')
    ax.set_ylabel('b (Angstrom)')
    ax.set_zlabel('c (Angstrom)')
    # Set the initial view
    ax.view_init(30, angle)
    # Hide the axes
    
    plt.show()
    
    return

def create_crystal_graph_central(molecule_interactions,consider_interactions='all'):
    all_mol_interactions = molecule_interactions.set_index(['mol1s','mol2s'])
    mol_interactions = all_mol_interactions.dropna(axis=0,how='any')
    idx1 = mol_interactions.index.get_level_values(0)[-1]
    g = nx.Graph()
    g.add_node(idx1,pos=np.array([0,0,0]))
    for idx in mol_interactions.index:
        disp = mol_interactions.loc[idx,['a','b','c']].values
        if idx[0] != idx1:
            disp = disp*-1
        dist = mol_interactions.at[idx,'dists']
        if consider_interactions == 'all':
            interactions = mol_interactions.loc[idx,['vdw_contact','hydrogen_bond','pi_pi_bond',
                                                         'halogen_bond','ch_pi_bond','hydrophobic_cc_bond']]
        else:
            interactions = mol_interactions.loc[idx,consider_interactions]
        angle = mol_interactions.at[idx,'interplanar_angles']
        if np.sum(interactions.values) > 0 :
            if idx[1] not in g:
                g.add_node(idx[1],pos=disp)
            if idx[0] not in g:
                g.add_node(idx[0],pos=disp)
            info = interactions.to_dict()
            info.update({'angle':np.round(angle,-1),'dist':np.round(dist,3)})
            g.add_edge(idx[0],idx[1],info=info)      
            
    return g

def create_crystal_graph(molecule_interactions,consider_interactions='all'):
    # Add interactions between all relevant molecules. Only calculated for central molecules before  
    mol_interactions = molecule_interactions.set_index(['mol1s','mol2s'])
    index = mol_interactions.index
    contact_info = mol_interactions.dropna(axis=0,how='any')
    contact_info = contact_info.filter(['dists','vdw_contact','hydrogen_bond','pi_pi_bond',
                         'halogen_bond','ch_pi_bond','hydrophobic_cc_bond'])
    contact_info.drop_duplicates('dists',keep='first',inplace=True)
    mol_interactions.drop(['vdw_contact','hydrogen_bond','pi_pi_bond',
                            'halogen_bond','ch_pi_bond','hydrophobic_cc_bond'],inplace=True,axis=1)
    mol_interactions = pd.merge(mol_interactions,contact_info,on='dists',how='left')
    mol_interactions.fillna(0,inplace=True)
    mol_interactions.set_index(index,inplace=True)
    # Create Crystal Graph
    g = nx.Graph()
    idx1 = molecule_interactions.mol1s[0]
    g.add_node(idx1,pos=np.array([0,0,0]))
    for idx in mol_interactions.index:
        disp = mol_interactions.loc[idx,['a','b','c']].values
        dist = mol_interactions.at[idx,'dists']
        if consider_interactions == 'all':
            interactions = mol_interactions.loc[idx,['vdw_contact','hydrogen_bond','pi_pi_bond',
                                                         'halogen_bond','ch_pi_bond','hydrophobic_cc_bond']]
        else:
            interactions = mol_interactions.loc[idx,consider_interactions]
        angle = mol_interactions.at[idx,'interplanar_angles']
        if idx[1] not in g:
            g.add_node(idx[1],pos=disp)
        if ((np.sum(interactions.values) > 0)):
            info = interactions.to_dict()
            info.update({'angle':np.round(angle,-1),'dist':np.round(dist,3)})
            g.add_edge(idx[0],idx[1],info=info)      
            
    return g