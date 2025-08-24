import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

# Generate the adjacency matrix
def get_adjacency_matrix(G):
    # Convert the graph to a sorted list of nodes for consistent indexing
    nodes = sorted(G.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    
    # Create an empty adjacency matrix
    size = len(nodes)
    adjacency_matrix = np.zeros((size, size), dtype=int)
    
    # Fill the matrix based on edges in the graph
    for edge in G.edges():
        i, j = node_index[edge[0]], node_index[edge[1]]
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1  # Ensure symmetry for undirected edges
    
    return adjacency_matrix, nodes

def initialize_graph(num_input_qubits):
    """
    Initialize the graph with N input qubits and 1 central output qubit.
    """
    G = nx.Graph()
    input_qubits = [f"I{i}" for i in range(num_input_qubits)]
    G.add_nodes_from(input_qubits)
    
    # Add central output qubit
    central_output = "O1"
    G.add_node(central_output)
    
    # Connect input qubits to the central output qubit
    for iq in input_qubits:
        G.add_edge(iq, central_output)
    
    return G, [central_output]  # Return the graph and the initial output qubit

def expand_output_qubit(G, output_qubit, step):
    """
    Expand an output qubit into 3 interconnected qubits (triangle) and redistribute connections.
    """
    # Identify neighbors of the output qubit
    neighbors = list(G.neighbors(output_qubit))
    
    # Remove the output qubit from the graph
    G.remove_node(output_qubit)
    
    # Add three new qubits in place of the removed output qubit
    new_qubits = [f"{output_qubit}_{step}_1", f"{output_qubit}_{step}_2", f"{output_qubit}_{step}_3"]
    G.add_nodes_from(new_qubits)
    
    # Connect the three new qubits in a triangle (periodic boundary conditions)
    G.add_edges_from([(new_qubits[0], new_qubits[1]),
                    (new_qubits[1], new_qubits[2]),
                    (new_qubits[2], new_qubits[0])])
    
    # Redistribute the connections to the new qubits
    random.shuffle(neighbors)
    for idx, neighbor in enumerate(neighbors):
        # Assign each neighbor to one of the new qubits
        target_qubit = new_qubits[idx % 3]
        G.add_edge(neighbor, target_qubit)
    
    return new_qubits


def expand_graph(G, initial_output_qubits, steps):
    """
    Perform the expansion process for a defined number of steps.
    """
    current_output_qubits = initial_output_qubits
    for step in range(1, steps + 1):
        # Choose one of the current output qubits to expand
        chosen_output = random.choice(current_output_qubits)
        
        # Expand the chosen output qubit
        new_qubits = expand_output_qubit(G, chosen_output, step)
        
        # Update the list of current output qubits
        current_output_qubits.remove(chosen_output)
        current_output_qubits.extend(new_qubits)
    
    return G

def expand_graph_and_plot(G, initial_output_qubits, steps):
    """
    Perform the expansion process for a defined number of steps and plot each step.
    """
    current_output_qubits = initial_output_qubits
    for step in range(1, steps + 1):
        # Choose one of the current output qubits to expand
        chosen_output = random.choice(current_output_qubits)
        
        # Expand the chosen output qubit
        new_qubits = expand_output_qubit(G, chosen_output, step)
        
        # Update the list of current output qubits
        current_output_qubits.remove(chosen_output)
        current_output_qubits.extend(new_qubits)
        
        # Plot the graph at this step
        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_size=500, font_size=10)
        plt.title(f"Step {step}: Expanded {chosen_output}")
        plt.show()

# # Example Usage
# N = 3  # Number of input qubits
# steps = 50  # Number of expansion steps

# # Initialize the graph
# G, initial_output_qubits = initialize_graph(N)

# # Perform the expansion and plot each step
# expand_graph_and_plot(G, initial_output_qubits, steps)