import os
import networkx as nx
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

def load_graphs_from_directory(directory):
    """
    Load all NetworkX graphs saved in pickle format in a directory.

    Parameters:
    directory (str): The path to the directory containing the pickle files.

    Returns:
    list: A list of NetworkX graphs loaded from the directory.

    """
    graphs = []

    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            with open(os.path.join(directory, filename), 'rb') as f:
                graph = pickle.load(f)
                if isinstance(graph, nx.Graph):
                    graphs.append((graph, int(filename.split(".")[0].split("_")[-1])))

    return graphs



def nx_to_pyg(graph, label):
    """Function transferring the given networkx graph together with associated label to
    a pytorch geometric Data object.

    Args:
        graph (networkx.Graph): networkx graph object for cancer prediction
        label (int): label of the graph, 0 (no cancer) or 1 (cancer)

    Returns:
        torch_geometric.data: pytorch geometric data representation of the graph
    """
    # Get feature names from first node
    feature_names = list(next(iter(graph.nodes(data=True)))[1].keys())

    # Extract node features
    node_features = []
    for node in graph.nodes(data=True):
        features = [node[1][feature_name] for feature_name in feature_names]
        node_features.append(features)

    # Convert lists to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long)

    # Extract edge indices and edge weights
    edge_index = []
    edge_weights = []
    for edge in graph.edges(data=True):
        edge_index.append((edge[0], edge[1]))
        edge_weights.append(edge[2]['weight'])
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    # Return the PyTorch Geometric data object
    return Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weights)


def nx_list_to_pyg(graph_list):
    """Function that takes a list of networkx graphs with labels and transfers all of them to 
    pytorch geometric Data objects. Uses nx_to_pyg function to do that.

    Args:
        graph_list (list): list of tuples, where first tuple element is the networkx graph and the second one is the cancer label (int)

    Returns:
        list: list of torch_geometric.data objects containing the graphs
    """
    return [nx_to_pyg(graph, label) for graph, label in graph_list]
    
