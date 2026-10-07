import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx
from itertools import permutations
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt



n = 4
m = 3


atlas = nx.graph_atlas_g()

c = 0
gs = []
for G in atlas:
    if G.number_of_nodes()==n and G.number_of_edges()==m:
        c=c+1
        print(G)
        gs.append(G)

print("atlas length",len(atlas))




def custom_edge_del(edge_attr):
    return 0.7071/2.0

def custom_edge_ins(edge_attr):
    return 0.7071/2.0




'''
Draw each isomorphism class based on n and m
'''

def save_fig(index):
    # 1. Initialize an empty undirected graph object
    G = gs[index]


    pos = nx.circular_layout(G)

    nx.draw(G, pos, with_labels=False, node_color='skyblue',node_size=800, font_size=12, font_weight='bold', edge_color='gray')
    caption_text = "Isomorphism Class"+str(index+1)
    plt.figtext(0.5, 0.01, caption_text, wrap=True, horizontalalignment='center', fontsize=10)
    plt.savefig("/kaggle/working/G_"+str(n)+"_"+str(m)+ "_"+str(index+1)+".png")
    plt.clf()


for i in range(1,len(gs)+1):
    save_fig(i-1)


def centered_simplex(n):
    """
    Return the n x n matrix whose columns are the vertices
    of the centered regular simplex.
    """
    I = np.eye(n)
    J = np.ones((n, n)) / n
    return I - J


def embedding_matrix(G, V):
    """
    Construct X(G) = [V | E], where E contains the
    midpoint vectors of the graph edges.

    Vertices of G must be labeled 0,...,n-1.
    """
    n = len(G)

    edge_points = []

    for i, j in G.edges():
        e = (V[:, i] + V[:, j]) / 2.0
        edge_points.append(e)

    if edge_points:
        E = np.column_stack(edge_points)
    else:
        E = np.empty((n, 0))

    return np.column_stack((V, E))


def permute_graph(G, perm):
    """
    Relabel G according to

        i -> perm[i]

    where perm is a tuple/list of length n.
    """
    mapping = {i: perm[i] for i in range(len(perm))}
    return nx.relabel_nodes(G, mapping)


def orthogonal_procrustes(A, B):
    """
    Solve

        min_M ||M A - B||_F

    subject to M^T M = I.

    A and B are n x k matrices.

    Returns
    -------
    M : n x n orthogonal matrix
    error : Frobenius norm ||M A - B||_F
    """

    # We need to maximize tr(M A B^T).
    C = A @ B.T

    U, S, Vt = np.linalg.svd(C)

    M = Vt.T @ U.T

    error = np.linalg.norm(M @ A - B, 'fro')

    return M, error


def GGD_a(G1, G2):
    """
    Compute the exact exhaustive-search version of GGD_a.

    GGD_a(G1,G2) =
        min_{M in O(n), P in P_str}
            || M X(G1) - X(G2) P ||_F

    Assumptions:
        - G1 and G2 are simple graphs
        - same number of vertices
        - same number of edges
        - vertices are labeled 0,...,n-1

    Returns
    -------
    best_distance
    best_permutation
    best_M
    """

    n = len(G1)

    if len(G2) != n:
        raise ValueError("Graphs must have the same number of vertices.")

    if G1.number_of_edges() != G2.number_of_edges():
        raise ValueError("Graphs must have the same number of edges.")

    # Standard centered simplex
    V = centered_simplex(n)

    # Embedding of G1
    X1 = embedding_matrix(G1, V)

    best_distance = np.inf
    best_perm = None
    best_M = None

    # Exhaustive search over vertex permutations
    for perm in permutations(range(n)):

        # Relabel G2 according to the vertex permutation
        G2p = permute_graph(G2, perm)

        # Construct the corresponding embedding
        X2p = embedding_matrix(G2p, V)

        # For this fixed structured permutation P,
        # solve the orthogonal Procrustes problem.
        M, distance = orthogonal_procrustes(X1, X2p)

        if distance < best_distance:
            best_distance = distance
            best_perm = perm
            best_M = M

    return best_distance
    

'''
Implementation of GGD_b
'''

def midpoint_cloud(G, V):

    pts=[]

    for u,v in G.edges():

        pts.append((V[:,u]+V[:,v])/2)

    return np.array(pts)

def cloud_distance(P,Q):

    
    m=len(P)

    D=np.zeros((m,m))

    for i in range(m):
        for j in range(m):

            cost = np.linalg.norm(P[i]-Q[j])
            
            D[i,j]=cost

    rows,cols=linear_sum_assignment(D)

    return D[rows,cols].sum()




def GGD_b(G1,G2):


    c=0
    n=G1.number_of_nodes()

    if n!=G2.number_of_nodes():
        raise ValueError("Different number of vertices")

    if G1.number_of_edges()!=G2.number_of_edges():
        raise ValueError("Different number of edges")

    V=centered_simplex(n)

    cloud1=midpoint_cloud(G1,V)

    best=np.inf
    
    for perm in permutations(range(n)):

        H=nx.relabel_nodes(
            G2,
            {i:perm[i] for i in range(n)}
        )

        cloud2=midpoint_cloud(H,V)

        d=cloud_distance(cloud1,cloud2)

        if(d<0.00001):
            c=c+1
        
        if d<best:
            best=d
        
    
    return best


# Creating the table of results
for i in range(0,len(gs)):
    GA  = []
    GB =  []
    GED = []
    for j in range(0,len(gs)):
        ga_ij= round(GGD_a(gs[i],gs[j]),4)  
        gb_ij= round(GGD_b(gs[i],gs[j]) ,4) 
        ged_ij=round(nx.graph_edit_distance(gs[i],gs[j],
        edge_del_cost=custom_edge_del,
        edge_ins_cost=custom_edge_ins),4)

        print("",ga_ij ,gb_ij, ged_ij, " &", end='')

    print()
