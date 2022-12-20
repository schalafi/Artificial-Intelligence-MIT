# Fall 2012 6.034 Lab 2: Search
#
# Your answers for the true and false questions will be in the following form.  
# Your answers will look like one of the two below:
#ANSWER1 = True
#ANSWER1 = False

# 1: True or false - Hill Climbing search is guaranteed to find a solution
#    if there is a solution
ANSWER1 = False 

# 2: True or false - Best-first search will give an optimal search result
#    (shortest path length).
#    (If you don't know what we mean by best-first search, refer to
#     http://courses.csail.mit.edu/6.034f/ai3/ch4.pdf (page 13 of the pdf).)
ANSWER2 = False

# 3: True or false - Best-first search and hill climbing make use of
#    heuristic values of nodes.
ANSWER3 = True

# 4: True or false - A* uses an extended-nodes set.
ANSWER4 = True

# 5: True or false - Breadth first search is guaranteed to return a path
#    with the shortest number of nodes.
ANSWER5 = True

# 6: True or false - The regular branch and bound uses heuristic values
#    to speed up the search for an optimal path.
ANSWER6 = False 

# Import the Graph data structure from 'search.py'
# Refer to search.py for documentation
from search import Graph

## Optional Warm-up: BFS and DFS
# If you implement these, the offline tester will test them.
# If you don't, it won't.
# The online tester will not test them.

def bfs(graph, start, goal):
    """
    graph: Graph 
    start: str 
        name of start node 
    goal: str 
        name of goal node
    It uses a queue to traverse the graph
    Return a path to the goal (list)
    """
    Q = [start] # Queue
    visited = set(start)
    parents = {}
    
    while len(Q) > 0: 
        u =   Q.pop(0)
        neighbors = graph.get_connected_nodes(u)
        for n in neighbors:
            if n not in visited:
                if n == goal:
                    parents[goal] = u 
                    path = get_path(parents,goal)
                    return path 
                parents[n] = u 
                visited.add(n)
                Q.append(n)

    return [start]

def get_path(parents,goal):
    """
    parents: dict str:str
    goal: str
    """
    path = [goal]
    
    if len(parents) ==0:
        return path

    next_node = goal 
    
    while next_node != None:
        node = parents.get(next_node,None)
        path.append(node)
        next_node = node

        if len(path) > 100:
            print("path is too long: ", path)
            return path 

        
    path.reverse()

    if path[0] is None:
        path = path[1:]

    return  path 

## Once you have completed the breadth-first search,
## this part should be very simple to complete.
def dfs(graph, start, goal):
    """
    graph: Graph 
    start: str 
        name of start node 
    goal: str 
        name of goal node
    It uses a stack to traverse the graph
    Return a path to the goal (list)
    """
    Q = [start] # Stack LIFO
    visited = set(start)
    parents = {}

    while len(Q)>0:
        u = Q.pop()
        for n in graph.get_connected_nodes(u):
            if n  not in visited:
                if n == goal:
                    parents[n] =  u
                    return get_path(parents, goal)
                Q.append(n)
                visited.add(n)
                parents[n] = u 

    return [start]

## Now we're going to add some heuristics into the search.  
## Remember that hill-climbing is a modified version of depth-first search.
## Search direction should be towards lower heuristic values to the goal.
def hill_climbing(graph, start, goal):
    """
    graph: Graph 
    start: str 
        name of start node 
    goal: str 
        name of goal node
    It uses a stack to traverse the graph
    Return a path to the goal (list)
    """
    Q = [start] # Stack LIFO
    visited = set(start)
    parents = {}

    #compute heristic (number of edges from node to goal)
    #H = heuristic_distance(graph,goal)
    #print("Heuritic distance")
    #print(H)

    while len(Q)>0:
        u = Q.pop()
        #Check for goal
        if u == goal:    
            return get_path(parents, goal)

        #tentative children
        #check for loops 
        maybe_children = graph.get_connected_nodes(u)

        #reject paths with loops 
        # node in maybe_children but not been visited
        children=list(set(maybe_children)-visited)

        #sort the new paths  according to the heuristic (function).
        #from max to min 
        children.sort(
            key = lambda n : graph.get_heuristic(n,goal),
            reverse = True)

        for c in children:
            parents[c] =  u
            #check path length
            Q.append(c)
            visited.add(c)
    return [start]

def heuristic_distance(graph,goal):
    """
    Compute the distance (number of edges) from each  node 
    to the goal 
    Return a dictionary of {node: distance to goal }
    """

    distances = {}

    for node in graph.nodes:
        distance = len(dfs(graph, node, goal)) -1
        if distance == -1:
            distance = float('inf')
        distances[node] = distance
    
    return distances 



# Import the Queue class from the queue module
from queue import Queue

def beam_searchGPT2(start_node, expand_fn, beam_width):
    # Initialize an empty queue of nodes to explore
    queue = Queue()

    # Add the starting node to the queue
    queue.put(start_node)

    # Loop until the queue is empty
    while not queue.empty():
        # Initialize an empty list of the best nodes at this step
        best_nodes = []

        # Loop through the nodes in the queue
        for _ in range(queue.qsize()):
            # Remove the next node from the queue
            node = queue.get()

            # Expand the node by generating its children
            children = expand_fn(node)

            # Add the children to the queue
            for child in children:
                queue.put(child)

            # Add the node to the list of the best nodes at this step
            best_nodes.append(node)

        # Sort the list of best nodes in descending order by their score
        best_nodes.sort(key=lambda node: node.score, reverse=True)

        # Keep only the top K nodes, where K is the beam width
        for node in best_nodes[:beam_width]:
            queue.put(node)

    # Return the list of best nodes found by the search
    return best_nodes


def beam_searchGPT(graph, start, goal, beam_width):

    # Initialize an empty queue of nodes to explore
    queue = []
    parents = {}
    expanded = set()

    # Add the starting node to the queue
    queue.append(start)

    # Loop until the queue is empty
    while len(queue) != 0:
        # Initialize an empty list of the best nodes at this step
        best_nodes = []

        # Loop through the nodes in the queue
        for node in queue:
            expanded.add(node)
            
            if node == goal:
                return get_path(parents,goal)   
            # Expand the node by generating its children
            children = graph.get_connected_nodes(node)

            
            for c in children:
                if c not in expanded:
                    parents[c]= node 
                    # Add the children to the queue
                    queue.append(c)

            # Add the node to the list of the best nodes at this step
            best_nodes.append(node)

        # Sort the list of best nodes in descending order by their score
        best_nodes.sort(key = lambda node: graph.get_heuristic(node,goal), reverse=True)

        # Keep only the top K nodes, where K is the beam width [min heristic, ..., max heuristic]
        queue = best_nodes[:beam_width]
    
    return []

#https://www.baeldung.com/cs/beam-search
## Now we're going to implement beam search, a variation on BFS
## that caps the amount of memory used to store paths.  Remember,
## we maintain only k candidate paths of length n in our agenda at any time.
## The k top candidates are to be determined using the 
## graph get_heuristic function, with lower values being better values.
def beam_search(graph, start, goal, beam_width):
    """
    graph: Graph 
    start: str 
        name of start node 
    goal: str 
        name of goal node
    beam_width: int 
        number of best paths to keep.
    It uses a stack to traverse the graph
    Return a path to the goal (list)
    """
    #Like BFS
    Q = [start] # Stack FIFO
    visited = set(start)
    parents = {}

    while len(Q) > 0: 
        u =   Q.pop(0)
        if u == goal:
            return get_path(parents,goal)

        neighbors = graph.get_connected_nodes(u)

        for n in neighbors:
            if n not in visited:
                parents[n] = u
                visited.add(n)
                Q.append(n)
        #min to max
        Q.sort(key = lambda node: graph.get_heuristic(node,goal) + ord(node) )

        if len(Q)>beam_width:
            Q =Q[:beam_width]
        #print("Q beam search:")
        #print(Q)
        #print( 
        #    list(map(lambda node: graph.get_heuristic(node,goal),Q ) )
        #    )
        #print()

    return []



"""def beam_search_distance(graph, start, goal, beam_width):
    "
    graph: Graph 
    start: str 
        name of start node 
    goal: str 
        name of goal node
    beam_width: int 
        number of best paths to keep.
    It uses a stack to traverse the graph
    Return a path to the goal (list)
    "
    Q = [start] # Stack LIFO
    visited = set()
    parents = {}

    while len(Q)>0:
        #print("Q: ",Q )
        u = Q.pop()
        visited.add(u)

        if u == goal:
            return get_path(parents,goal)

        children = graph.get_connected_nodes(u)
        # sort the neighbors according to the heuristic (function).
        #from max to min 
        children.sort(key = lambda n : graph.get_heuristic(n,goal), reverse = True) 
        m = len(children)

        for i in range(m) :

            #keep only the beam with best candidates
            node  = children[m-(i+1)] 

            if node not in Q and  node not in visited:
                Q.extend([node]) 
            
            elif node in Q:
                current_path = get_path(parents, goal)

                parent = parents.get(node, None)
                parents[node] = u 
                new_path = get_path(parents,goal)

                #if new path length is greater than current path length
                #conserve current path
                if path_length(graph, new_path) > path_length(graph,current_path):
                    parents[node] = parent
            
            elif node not in visited:
                Q.append(node)
        #Don't need to sort again
        #Q.sort(key = lambda n : graph.get_heuristic(n,goal), reverse = True)
        Q.reverse()

        if len(Q) > beam_width:
            #Keep only the best beam_width candidates
            Q = Q[-beam_width:]

    return []
"""
## Now we're going to try optimal search.  The previous searches haven't
## used edge distances in the calculation.

## This function takes in a graph and a list of node names, and returns
## the sum of edge lengths along the path -- the total distance in the path.
def path_length(graph, node_names):

    if len(node_names) <=1:
        return 0 
    cost = 0 
    previous= node_names[0]
    current = None 
    
    for node in node_names[1:]:
        current = node 
        cost+= graph.get_edge(previous, current).length
        previous  = current
    
    return cost 

def branch_and_bound(graph, start, goal):
    #stack LIFO 
    Q = [start]
    visited = set(start)
    parents = {} #{node: parent (node) }

    while len(Q)>0:
        u = Q.pop() 
        if u == goal:
            return get_path(parents, goal)
        
        children = graph.get_connected_nodes(u)

        for node in children:
            #reject loops
            if node not in visited:
                parents[node] =  u
                if node == goal:
                    return get_path(parents, goal)

                visited.add(node)
                Q.extend([node])
        #[max distance, ..., min distance]
        Q.sort(key = lambda n: path_length(graph, get_path(parents,n)), reverse = True )

    return []

def a_star(graph, start, goal):

    Q = [start]
    visited = set(start)
    parents = {}

    distance = { node: float('infinity') for node in graph.nodes}
    distance[start] = 0 

    f = {node: float('infinity') for node in graph.nodes}
    f[start] = graph.get_heuristic(start,goal)

    while len(Q) >0:
        u = Q.pop()
        if u == goal:
            return get_path(parents, goal)

        for node in graph.get_connected_nodes(u):
            if node not in visited:
                parents[node]= u 
                g = distance[u] + graph.get_edge(u,node).length

                if (g <= distance[node] ):
                    distance[node] = g
                    f[node] = g + graph.get_heuristic(node,goal)
                    visited.add(node)

                    if node not in Q:
                        Q.append(node)
        Q.sort(key = lambda node: f[node], reverse = True ) 
    return [start]





    


## It's useful to determine if a graph has a consistent and admissible
## heuristic.  You've seen graphs with heuristics that are
## admissible, but not consistent.  Have you seen any graphs that are
## consistent, but not admissible?

def is_admissible(graph, goal):
    raise NotImplementedError

def is_consistent(graph, goal):
    raise NotImplementedError

HOW_MANY_HOURS_THIS_PSET_TOOK = ''
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
