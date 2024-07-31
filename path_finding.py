
#! A* functions ----------------------------------------------------

# if a cell is given, return the 4-connected neighbors
# probability  high -----> cost low -----> high priority
def get_neighbors(grid, cell, map_height, map_width):

    neighbors = []
    x, y = cell

    # Check if the neighbor is within the map and is free
    if x > 0 and grid[x - 1,y] != 0:
        neighbors.append((x - 1, y))
    if x < map_width - 1 and grid[x + 1,y] != 0:
        neighbors.append((x + 1, y))
    if y > 0 and grid[x, y - 1] != 0:
        neighbors.append((x, y - 1))
    if y < map_height - 1 and grid[x, y + 1] != 0:
        neighbors.append((x, y + 1))

    return neighbors

def cost(current_cell, neighbor, grid):
    return 0.1 + round(1 - grid[neighbor],2)

def heuristic(current_cell, end_cell):
    return abs(current_cell[0] - end_cell[0]) + abs(current_cell[1] - end_cell[1])

#! A* algorithm ----------------------------------------------------

class node:

    childs = []
    parent = None
    cost = 0
    name = ""
    value = None

    #Constructor
    def __init__(self, value=None, childs=[], parent=None, cost=0, name=None):

        self.childs = childs
        self.value = value
        self.name = name
        self.parent = parent
        self.cost = cost

    #print function / string representation
    def __repr__(self) -> str:

        if(self.name):
            return self.name
        else:
            return str(self.value)
        
    #equality function
    def __eq__(self, o: object) -> bool:
        return self.value == o.value

def astar(start_cell, end_cell, grid, map_height, map_width):

    unvisited = []
    visited = []

    # Create the start node
    start_node = node(start_cell, cost=0)

    # Add the start node to the unvisited list
    unvisited.append(start_node)

    while unvisited:

        # Get the node with the minimum cost
        current_node = min(unvisited, key=lambda x: x.cost)

        # Remove the current node from the unvisited list
        unvisited.remove(current_node)

        # Add the current node to the visited list
        visited.append(current_node)

        # Check if the current node is the end node
        if current_node.value == end_cell:
            path = []
            while current_node:
                path.append(current_node.value)
                current_node = current_node.parent
            return path[::-1]

        # Get the neighbors of the current node
        neighbors = get_neighbors(grid, current_node.value, map_height, map_width)

        for neighbor in neighbors:

            # Create the neighbor node
            neighbor_node = node(neighbor, parent=current_node, cost=current_node.cost + cost(current_node.value, neighbor, grid) + heuristic(neighbor, end_cell))

            # Check if the neighbor is in the visited list
            if neighbor_node in visited:
                continue

            # Check if the neighbor is in the unvisited list
            if neighbor_node not in unvisited:
                unvisited.append(neighbor_node)
            else:
                # Replace the neighbor node with the new node if the new node has a lower cost
                for node_ in unvisited:
                    if node_ == neighbor_node and node_.cost > neighbor_node.cost:
                        node_ = neighbor_node
    return None
