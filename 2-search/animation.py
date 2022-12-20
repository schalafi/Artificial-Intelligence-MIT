from manim import *
import  networkx as nx 
# importing matplotlib.pyplot
import matplotlib.pyplot as plt

import search 
from graphs import NEWGRAPH1


default_color = WHITE
frontier_color = RED 

#Labeled edge
#It has weigtht 
class LabeledEdge(Line):
    def __init__(self,start=LEFT, end=RIGHT,weight=None, buff=0, path_arc=None, **kwargs):
        super().__init__(start=start, end=end, buff=buff, path_arc=path_arc, **kwargs)  

        self.weight = weight
        if weight is not None:
            assert  (type(weight) == int or type(weight) == float ), f"weight: {weight}  is not a number"
            self.label = MathTex(self.weight)
            self.label.move_to(self.get_center()+ Y_AXIS*self.label.height*3/4)
            self.add(self.label)
            
        
        
class TestLabeledEdge(Scene):
    def construct(self):

        le = LabeledEdge(weight = 10, start = LEFT, end = RIGHT)

        self.play(Create(le))

        self.wait(1)

        le2 = LabeledEdge(weight = 7, start = np.array([3,0,0]), end = np.array([6,6,0]))

        self.play(Create(le2))

        self.wait(1)

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

class SearchTree(MovingCameraScene):
    CONFIG = {
        'search_args':{
            'graph:':NEWGRAPH1,
            'start':'S',
            'goal':'G'
        }
    }

    def construct(self):
        #digest_config(self,self.CONFIG)
        nx_G = mit_to_nx(NEWGRAPH1)
        edges_attr = nx.get_edge_attributes(nx_G,'weight')

        G =Graph(list(nx_G.nodes),
            list(nx_G.edges),
            labels = True,
            label_fill_color  = BLACK,
            layout_scale=5,
            edge_type=LabeledEdge,
            edge_config={edge:{'weight': edges_attr[edge]} for edge in edges_attr.keys() }
            ) # Graph.from_networkx(nx_G)

        self.G = G
        self.camera.frame.set_height(G.height*1.1)
        self.camera.frame.move_to(G)
        self.play(Create(G))
        self.wait(2)

        self.a_star(NEWGRAPH1,'S','G')

        self.beam_search(NEWGRAPH1,'S','G',2)
        
        self.wait()

        #self.a_star()

    def change_color_node(self,node, color ):
        """
        node: hasable 
            name of the node in self.G
        color: str
            hexadecimal color 
        Change color of a node in self.G 
        
        """
        if color == BLACK:
            print("WARNING node and label will have the same color.")
        self.G[node].set_color(color) 
        self.G._labels[node].set_color(BLACK)


    def set_edges_opacity(self,opacity):
        """
        opacity: float
            Opacity in [0,1]
        Set the opacity of all edges
        """
        G = self.G
        #Make edges transparent to highlight the path 
        for edge in G.edges.keys():
            obj  = G.edges[edge]
            #self.play(obj.animate.set_fill(WHITE,opacity=0.1), run_time = 0.1)
            obj.save_state()
            obj.set_opacity(opacity)

    def show_path(self, path):
        """
        G: manim graph 
        path: list of  nodes (strings)
        Animate path creation
        """
        G = self.G

        #Make edges transparent to highlight the path 
        for edge in G.edges.keys():
            obj  = G.edges[edge]
            #self.play(obj.animate.set_fill(WHITE,opacity=0.1), run_time = 0.1)
            obj.save_state()
            obj.set_opacity(0.15)

        previous = path[0]
        for i in range(1,len(path)):
            next = path[i]
            edge_tuple = (previous, next)
            #print("edge_tuple: ", edge_tuple)
            edge  = G.edges.get(edge_tuple,None)
            #Try for (previous, next) and (next, previous)
            if edge is None:
                edge_tuple = (next, previous)
                #print("edge_tuple: ", edge_tuple)
                edge = G.edges.get(edge_tuple, None)
            
            edge.set_opacity(1)
            edge.set_color(BLUE_C)
            self.wait(1.0)
            previous = next 
        

        self.wait(5)
        #Restore edges to original state
        for edge in G.edges.keys():
            obj  = G.edges[edge]
            obj.restore()
    
    def draw_paths(self,parents, nodes = None ):
        """
        parents: Dict
        nodes: List of strings
            nodes whose paths will be shown

        """
        #Save original state of edges
        for (edge,object) in self.G.edges.items():
            object.save_state()
        G =self.G 
        path_edge_color = YELLOW
        #First get all paths form source
        #to  last node in the path   DIFFICULT! D:
        paths = []

        def draw_edge(node):
            parent = parents.get(node, None)
            if parent is None:
                return
            #try edge (parent,node)
            #or (node,parent) One of the two must exist in G
            edge = G.edges.get((parent,node) ,None ) 
            if edge is  None:
                edge = G.edges.get((node,parent) ,None )
            
            edge.set_color(path_edge_color)
            edge.set_opacity(1.0)
            self.wait(0.2)

        #Draw only the paths that end with the nodes
        if nodes is not None:
            for t_node in nodes:
                path = get_path(parents,t_node)
                for node in path:
                    draw_edge(node)

        else:
            #Draw each edge: Easy! :D
            for node in parents.keys():
                draw_edge(node)

        self.wait(1)

        #Restore original state of edges
        for (edge,object) in self.G.edges.items():
            object.restore()


    def update_Q(self,Q):
        """
        Q: list of strings
        Update the names in the frontier (Q) to the screen.

        """
        Q_elems = VGroup()

        for e in Q:
            name = Text(e)
            Q_elems.add(name)

        Q_elems.arrange(DOWN)
        Q_elems.to_edge(LEFT +UP)

        self.add(Q_elems)
        self.Q_elems = Q_elems
    
    def clear_Q(self):
        """
        Remove Q from scene.
        """
        self.remove(self.Q_elems)
    
    def show_score(self,f, score_symbol = r'f'):
        """
        f: Dict
        Show for each node his score
        can be f= distance + heuristic.
                or distance
                or heuristic or another function
        """
        for (key,value) in f.items():
            label = Tex(score_symbol +"= "+ str(value))
            label.set_color(BLUE_B)
            
            #Node is an object
            node = self.G[key]

            label.set_width(node.width*1.1)
            label.next_to(node,DOWN)
            #Add the label as a new attribute
            node.f = label

            self.add(label)

    def remove_score(self):
        nx_graph = self.G._graph
        nodes = nx_graph.nodes

        for node in nodes:
            #remove the f score from the scene
            self.remove(self.G[node].f)
        

    def update_f(self,node_name, f_value):
        """
        node_name: str
        f_value: float|int
        """
        node = self.G[node_name]
        new_label = Tex(r"f= " + str(f_value))
        new_label.move_to(node.f)
        new_label.match_style(node.f)
        new_label.match_width(node.f)

        node.f.become(new_label)

    
    def beam_search(self,graph, start, goal, beam_width):
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
        self.Q_elems = VGroup()
        #self.f_to_object = {node: }
        self.set_edges_opacity(0.10)
        run_time = 0.1

        #Get a mapping node: heuristic
        heuristic = {node: graph.get_heuristic(node,goal) for node in graph.nodes}
        self.show_score(heuristic, 'h')


        #Like BFS
        Q = [start] # Stack FIFO
        visited = set(start)
        parents = {}

        self.change_color_node(start, frontier_color)
        self.wait(run_time)

        self.update_Q(Q)

        while len(Q) > 0: 
            u =   Q.pop(0)
            self.clear_Q()
            self.update_Q(Q)
            self.change_color_node(u, default_color)
            self.wait(run_time)


            if u == goal:
                path = get_path(parents,goal)
                #Draw solution path 
                self.show_path(path)
                self.wait(1)
                self.remove_score()
                return path

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

            self.clear_Q()
            self.update_Q(Q)
            self.wait(run_time)
            for node in  [x  for x in graph.nodes if x  not in Q]:
                self.change_color_node(node, default_color)
                self.wait(run_time)    
            for  node in Q:
                self.change_color_node(node, frontier_color)
                self.wait(run_time)
            
            self.draw_paths(parents, Q) 
        self.remove_score()
        return []

    def a_star(self,graph, start, goal):
        self.Q_elems = VGroup()
        #self.f_to_object = {node: }
        self.set_edges_opacity(0.10)
    
        Q = [start]
        visited = set(start)
        parents = {}
        self.update_Q(Q)

        self.change_color_node(start, frontier_color)
        self.wait(0.1)

        distance = { node: float('infinity') for node in graph.nodes}
        distance[start] = 0 

        f = {node: float('infinity') for node in graph.nodes}
        f[start] = graph.get_heuristic(start,goal)

        self.show_score(f)

        while len(Q) >0:

            u = Q.pop()
            self.clear_Q()
            self.update_Q(Q)
            

            run_time = 0.1
            self.change_color_node(u, default_color)
            self.wait(run_time)


            if u == goal:
                path = get_path(parents, goal)
                print("Solution path: ", path)
                #Draw solution path 
                self.show_path(path)
                self.wait(1)
                self.remove_score()
                return path 

            for node in graph.get_connected_nodes(u):
                if node not in visited:
                    parents[node]= u 
                    g = distance[u] + graph.get_edge(u,node).length

                    if (g <= distance[node] ):
                        distance[node] = g
                        f[node] = g + graph.get_heuristic(node,goal)

                        #Update the f value in the scene
                        self.update_f(node,f[node])

                        visited.add(node)

                        if node not in Q:
                            Q.append(node)

                            self.clear_Q()
                            self.update_Q(Q)
                            self.change_color_node(node, frontier_color)
                            self.wait(run_time)

            self.draw_paths(parents)               
            Q.sort(key = lambda node: f[node], reverse = True ) 
        
        self.clear_Q()
        self.update_Q(Q)
        #Draw solution path 
        self.show_path([start])
        self.remove_score()
        return [start]


def mit_to_nx(G: search.Graph) -> nx.classes.graph.Graph:
    """
    G: Graph from seach.Graph into 
    nx.Graph
    return an instance of nx.Graph
    """
    nx_G = nx.Graph()
    nodes = G.nodes

    #add nodes
    nx_G.add_nodes_from(nodes)

    for edge in G.edges:
        nx_G.add_edge(edge.node1, edge.node2, weight =edge.length )

    return nx_G 

if __name__ == "__main__":
    print("nodes : ",NEWGRAPH1.nodes)
    print('edges:', NEWGRAPH1.edges)

    for edge in NEWGRAPH1.edges:
        print(edge)

    nx_G = mit_to_nx(NEWGRAPH1)
    print("edges and nodes:")
    print(nx_G.edges,nx_G.nodes)
    pos = nx.spring_layout(nx_G)
    nx.draw(nx_G,pos = pos, with_labels = True)
    #nx.draw_networkx_labels(nx_G)
    labels = nx.get_edge_attributes(nx_G,'weight')
    print("labels:" , labels)
    nx.draw_networkx_edge_labels(
        nx_G,
        pos = pos,
        edge_labels=labels)

    plt.savefig("NEWGRAPH1.png")
        