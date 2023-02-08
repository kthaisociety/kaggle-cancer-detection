import networkx as nx
import numpy as np
import pickle


def save_graph(G, filename):
    """
    Saves a networkx graph to a file in binary format.
    
    Parameters:
    G (networkx.Graph): The input graph.
    filename (str): The name of the file to save the graph to.
    
    Returns:
    None
    """
    pickle.dump(G, open(filename, 'wb'))



def generate_graph(features):
    """
    Generates a graph from the input image and segments using networkx.
    Each node corresponds to one segment.
    The edges weights are calculated based on similarity and distance of the segment centers.

    Parameters:
    features: List of dictionaries where each dictionary contains the features for one segment

    Returns:
    networkx.Graph: The generated graph with nodes and edges.
    """
    
    G = nx.Graph()
    
    G = generate_nodes(G, features)
    G = generate_edges(G)
    
    return G


def generate_nodes(G, features):
    """
    This function generates nodes for a networkx graph, G, with normalized features of the segments in an image.

    Parameters:
    G: A networkx graph to which nodes are added.
    image: An input image.
    features: List of dictionaries where each dictionary contains the features for one segment
    
    Returns:
    A networkx graph with nodes added, each node representing a segment in the image and having normalized features associated with it.
    """
    
    node_features = []

    for i, feature_set in enumerate(features):
        node_features.append((i, feature_set))
    
    G.add_nodes_from(node_features)

    return G
    

def generate_edges(G):
    """
    Generates edges in the networkx graph 'G' based on similarity between the feature vectors of nodes (excluding center_x and center_y) 
    and then multiplies the result with standardized distance (calculated with center_x and center_y as node coordinates).
    
    Parameters:
    G (networkx.Graph): The input graph.
    
    Returns:
    networkx.Graph: The graph with edges added.
    """
    threshold_distance = 10
    nodes = list(G.nodes(data=True))
    features = [list(node[1].keys()) for node in nodes] # each features entry is a list of the paraemter names for each node
    features = list(set(features[0]).intersection(*features)) # get common features accross all nodes
    features = [f for f in features if f not in ['center_x', 'center_y']] # exclude center_x and center_y and leave nly all oher features
    for i, (u, u_data) in enumerate(nodes[:-1]):
        
        standard_distance_array = []
        threshold_distance = 0
        
        for j, (v, v_data) in enumerate(nodes):
            x_distance = (u_data['center_x'] - v_data['center_x'])**2
            y_distance = (u_data['center_y'] - v_data['center_y'])**2
            standard_distance = np.sqrt(x_distance + y_distance)
            standard_distance_array.append(standard_distance)
        threshold_distance = np.sort(standard_distance_array)[int(np.size(standard_distance_array)*0.05)] # select closest 5% of neighbours
        
        for j, (v, v_data) in enumerate(nodes[i+1:]): # all elements after the i-th
            # local feature values for the two nodes
            u_features = np.array([u_data[f] for f in features]) 
            v_features = np.array([v_data[f] for f in features])
            # normalized dot product of the features
            feature_similarity = np.dot(u_features, v_features) / (np.linalg.norm(u_features) * np.linalg.norm(v_features))
            # sqrt(sx^2 + y^2) distance
            x_distance = (u_data['center_x'] - v_data['center_x'])**2
            y_distance = (u_data['center_y'] - v_data['center_y'])**2
            standard_distance = np.sqrt(x_distance + y_distance)
            standard_distance = standard_distance if standard_distance < threshold_distance else 0
            # weigth depends on distance and fearute dop product
            edge_weight = feature_similarity * standard_distance
            G.add_edge(u, v, weight=edge_weight)
            
    return G
