
import matplotlib.pyplot as plt
import random


class Point:
    def __init__(self, v, x, y):
        self.label = v
        self.x = x
        self.y = y


class Graph:
    def __init__(self):
        self.__graph_dict = {}
        self.__m = 0
        '''
        in order to plot the graph, for each node we define a position
        in a grid with a given number of columns
        '''
        self.__grid_cols = 3
        self.__positions = {}

    def nodeNumber(self):
        return len(self.__graph_dict)

    def edgeNumber(self):
        return self.__m

    def addNode(self, v):
        if v not in self.__graph_dict:
            self.__graph_dict[v] = []
            self.__updatePositions(v)

    def isEdge(self, u, v):
        if u in self.__graph_dict and \
                v in self.__graph_dict[u]:
            return True
        return False

    def addEdge(self, v0, v1):
        if not self.isEdge(v0, v1) and v0 != v1:
            self.__graph_dict[v0] = self.__graph_dict.get(v0, []) + [v1]
            self.__graph_dict[v1] = self.__graph_dict.get(v1, []) + [v0]
            self.__m += 1
            self.__updatePositions(v0)
            self.__updatePositions(v1)

    '''
    Returns the list of nodes that are neighbors  v
    '''

    def getNeighbors(self, v):
        if v in self.__graph_dict:
            return self.__graph_dict[v]

    '''
    define the position in the grid of node v. Called avery time a new
    node is added
    '''

    def __updatePositions(self, v):
        if v not in self.__positions:
            n = len(self.__positions)
            row = int(n / self.__grid_cols) + random.randint(-1, 1) / 5
            col = n % self.__grid_cols + random.randint(-1, 1) / 5
            self.__positions[v] = Point(v, col, row)

    def draw(self, c='b'):
        x, y = [], []
        for vertex in self.__positions:
            x.append(self.__positions[vertex].x)
            y.append(self.__positions[vertex].y)
        plt.scatter(x, y, c=c)

        for vertex in self.__positions:
            p = self.__positions[vertex]
            plt.annotate(p.label, (p.x, p.y))

        for v0 in self.__positions:
            edges_from_v0 = self.getNeighbors(v0)
            for v1 in edges_from_v0:
                p0 = self.__positions[v0]
                p1 = self.__positions[v1]
                plt.plot([p0.x, p1.x], [p0.y, p1.y], c=c)
        plt.show()


def findPath(g, start_v, end_v, visited=None):
    if start_v == end_v:
        return []

    if visited == None:
        visited = set([start_v])

    for x in g.getNeighbors(start_v):
        if x not in visited:
            visited.add(x)
            next_path = findPath(g, x, end_v, visited)
            if next_path != None:
                return [(start_v, x)] + next_path

    return None


'''
	returns true if there exist a path between v0 and v1 of size
	at most 2 ( at most two edges in the path)
'''


def path2(g, v0, v1):
    if g.isEdge(v0, v1):
        return True
    a = set(g.getNeighbors(v0))
    b = set(g.getNeighbors(v1))
    if len(a.intersection(b)) >= 1:
        return True
    return False


def isIndSet(g, I):
    n = len(I)
    for i in range(n - 1):
        for j in range(i + 1, n):
            x, y = I[i], I[j]
            if g.isEdge(x, y):
                return False
    return True


'''
Version 0: the computational cost is O(n*|I|^2)
where n = number of nodes in g
'''


def inducedSubGraph0(g, I):
    g1 = Graph()
    for x in I:
        for y in g.getNeighbors(x):
            if y in I:
                g1.addEdge(x, y)
    return g1


'''
A more efficient version, its computational cost is O(|I|^2)
'''


def inducedSubGraph(g, I):
    n = len(I)
    g1 = Graph()
    for i in range(n - 1):
        for j in range(i + 1, n):
            x, y = I[i], I[j]
            if g.isEdge(x, y):
                g1.addEdge(x, y)
    return g1


g = Graph()
g.addEdge('a', 'b')
g.addEdge('a', 'f')
g.addEdge('b', 'c')
g.addEdge('b', 'e')
g.addEdge('c', 'e')
g.addEdge('c', 'd')
g.addEdge('e', 'f')
g.addEdge('d', 'f')

g.addEdge('u', 'v')

print(path2(g, 'a', 'c'))

g.draw()

