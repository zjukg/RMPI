from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

graph = [[0, 1, 2, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 3],
         [0, 0, 0, 0]]

graph = csr_matrix(graph)
print(graph)

dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=0, return_predecessors=True, min_only=False)
print(dist_matrix)
print(dist_matrix.shape)