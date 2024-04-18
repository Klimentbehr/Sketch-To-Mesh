def create_edge_value_association(edge_points):
    edge_values = {}
    value_counter = 0
    
    for edge in edge_points:
        # Convert edge points to tuple for hashability
        edge_tuple = (tuple(edge[0]), tuple(edge[1]))
        edge_rev_tuple = (tuple(edge[1]), tuple(edge[0]))
        
        # Check if the edge already has an associated value
        if edge_tuple not in edge_values and edge_rev_tuple not in edge_values:
            edge_values[edge_tuple] = value_counter
            value_counter += 1
    
    # Create a list of values corresponding to each edge
    edge_value_list = [edge_values[(tuple(edge[0]), tuple(edge[1]))] for edge in edge_points]
    
    return edge_value_list

# Test usage:
edge_points = [
    [(5, 3), (7, 9)],
    [(2, 8), (6, 4)],
    [(0, 1), (3, 7)],
    [(9, 2), (5, 6)],
    [(9, 2), (5, 5)]
    
]

edge_values = create_edge_value_association(edge_points)
print(edge_values)
