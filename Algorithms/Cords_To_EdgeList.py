from dataclasses import dataclass, field
from typing import Dict, Tuple, List

# Define the data class for adjacent points
@dataclass
class AdjacentPoint:
    coordinate: Tuple[int, int]
    adjacent_line: Dict[Tuple[int, int], List[Tuple[int, int]]]

# Function to map a list of coordinate pairs to integer indices
def map_coordinates_to_indices(user_sequence):
    point_index = {}  # This dictionary will map points to integers
    index = 0
    indexed_pairs = []

    for pair in user_sequence:
        # Unpack the pair for clarity
        start, end = pair
        
        # Check if the start coordinate is new, if so, add to the point_index
        if start not in point_index:
            point_index[start] = index
            index += 1
            
        # Check if the end coordinate is new, if so, add to the point_index
        if end not in point_index:
            point_index[end] = index
            index += 1

        # Append the index pair to the output list
        indexed_pairs.append((point_index[start], point_index[end]))

    return indexed_pairs

# Example user input
user_sequence = [
    ((5, 3), (7, 9)), 
    ((7, 9), (2, 8)), 
    ((2, 8), (6, 4)), 
    ((6, 4), (5, 3))
]

# Get the output list of indexed pairs
indexed_edge_list = map_coordinates_to_indices(user_sequence)

print("Indexed Edge List:", indexed_edge_list)
