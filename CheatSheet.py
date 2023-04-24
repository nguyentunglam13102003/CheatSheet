# Dijkstra’s algorithm

class Graph():

	def __init__(self, vertices):
		self.V = vertices
		self.graph = [[0 for column in range(vertices)]
					for row in range(vertices)]

	def printSolution(self, dist):
		print("Vertex \t Distance from Source")
		for node in range(self.V):
			print(node, "\t\t", dist[node])

	def minDistance(self, dist, sptSet):

		min = 1e7

		for v in range(self.V):
			if dist[v] < min and sptSet[v] == False:
				min = dist[v]
				min_index = v

		return min_index

	def dijkstra(self, src):

		dist = [1e7] * self.V
		dist[src] = 0
		sptSet = [False] * self.V

		for cout in range(self.V):

			u = self.minDistance(dist, sptSet)

			sptSet[u] = True

			for v in range(self.V):
				if (self.graph[u][v] > 0 and
				sptSet[v] == False and
				dist[v] > dist[u] + self.graph[u][v]):
					dist[v] = dist[u] + self.graph[u][v]

		self.printSolution(dist)


g = Graph(9)
g.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
		[4, 0, 8, 0, 0, 0, 0, 11, 0],
		[0, 8, 0, 7, 0, 4, 0, 0, 2],
		[0, 0, 7, 0, 9, 14, 0, 0, 0],
		[0, 0, 0, 9, 0, 10, 0, 0, 0],
		[0, 0, 4, 14, 10, 0, 2, 0, 0],
		[0, 0, 0, 0, 0, 2, 0, 1, 6],
		[8, 11, 0, 0, 0, 0, 1, 0, 7],
		[0, 0, 2, 0, 0, 0, 6, 7, 0]
		]

g.dijkstra(0)


# Prim's Algorithm in Python


INF = 9999999

V = 5

G = [[0, 9, 75, 0, 0],
     [9, 0, 95, 19, 42],
     [75, 95, 0, 51, 66],
     [0, 19, 51, 0, 31],
     [0, 42, 66, 31, 0]]

selected = [0, 0, 0, 0, 0]

no_edge = 0

selected[0] = True

print("Edge : Weight\n")
while (no_edge < V - 1):

    minimum = INF
    x = 0
    y = 0
    for i in range(V):
        if selected[i]:
            for j in range(V):
                if ((not selected[j]) and G[i][j]):
                    if minimum > G[i][j]:
                        minimum = G[i][j]
                        x = i
                        y = j
    print(str(x) + "-" + str(y) + ":" + str(G[x][y]))
    selected[y] = True
    no_edge += 1

# Kruskal's Algorithm

class Graph:

	def __init__(self, vertices):
		self.V = vertices
		self.graph = []

	def addEdge(self, u, v, w):
		self.graph.append([u, v, w])

	def find(self, parent, i):
		if parent[i] != i:


			parent[i] = self.find(parent, parent[i])
		return parent[i]

	def union(self, parent, rank, x, y):

		if rank[x] < rank[y]:
			parent[x] = y
		elif rank[x] > rank[y]:
			parent[y] = x

		else:
			parent[y] = x
			rank[x] += 1


	def KruskalMST(self):


		result = []

		i = 0

		e = 0

		self.graph = sorted(self.graph,
							key=lambda item: item[2])

		parent = []
		rank = []

		for node in range(self.V):
			parent.append(node)
			rank.append(0)


		while e < self.V - 1:

			u, v, w = self.graph[i]
			i = i + 1
			x = self.find(parent, u)
			y = self.find(parent, v)


			if x != y:
				e = e + 1
				result.append([u, v, w])
				self.union(parent, rank, x, y)


		minimumCost = 0
		print("Edges in the constructed MST")
		for u, v, weight in result:
			minimumCost += weight
			print("%d -- %d == %d" % (u, v, weight))
		print("Minimum Spanning Tree", minimumCost)

if __name__ == '__main__':
	g = Graph(4)
	g.addEdge(0, 1, 10)
	g.addEdge(0, 2, 6)
	g.addEdge(0, 3, 5)
	g.addEdge(1, 3, 15)
	g.addEdge(2, 3, 4)

	g.KruskalMST()

#DFS Algorithm

from collections import defaultdict


class Graph:

	# Constructor
	def __init__(self):


		self.graph = defaultdict(list)


	def addEdge(self, u, v):
		self.graph[u].append(v)


	def DFSUtil(self, v, visited):


		visited.add(v)
		print(v, end=' ')

		for neighbour in self.graph[v]:
			if neighbour not in visited:
				self.DFSUtil(neighbour, visited)


	def DFS(self, v):


		visited = set()


		self.DFSUtil(v, visited)


if __name__ == "__main__":
	g = Graph()
	g.addEdge(0, 1)
	g.addEdge(0, 2)
	g.addEdge(1, 2)
	g.addEdge(2, 0)
	g.addEdge(2, 3)
	g.addEdge(3, 3)

	print("Following is DFS from (starting from vertex 2)")

	g.DFS(2)

#BFS Algorithm

from collections import defaultdict

class Graph:

	def __init__(self):


		self.graph = defaultdict(list)


	def addEdge(self, u, v):
		self.graph[u].append(v)

	def BFS(self, s):


		visited = [False] * (max(self.graph) + 1)


		queue = []


		queue.append(s)
		visited[s] = True

		while queue:


			s = queue.pop(0)
			print(s, end=" ")


			for i in self.graph[s]:
				if visited[i] == False:
					queue.append(i)
					visited[i] = True

if __name__ == '__main__':

	g = Graph()
	g.addEdge(0, 1)
	g.addEdge(0, 2)
	g.addEdge(1, 2)
	g.addEdge(2, 0)
	g.addEdge(2, 3)
	g.addEdge(3, 3)

	print("Following is Breadth First Traversal"
		" (starting from vertex 2)")
	g.BFS(2)

#Merge Sort

def merge(arr, l, m, r):
	n1 = m - l + 1
	n2 = r - m


	L = [0] * (n1)
	R = [0] * (n2)


	for i in range(0, n1):
		L[i] = arr[l + i]

	for j in range(0, n2):
		R[j] = arr[m + 1 + j]

	i = 0
	j = 0
	k = l

	while i < n1 and j < n2:
		if L[i] <= R[j]:
			arr[k] = L[i]
			i += 1
		else:
			arr[k] = R[j]
			j += 1
		k += 1


	while i < n1:
		arr[k] = L[i]
		i += 1
		k += 1

	while j < n2:
		arr[k] = R[j]
		j += 1
		k += 1


def mergeSort(arr, l, r):
	if l < r:

		m = l+(r-l)//2

		mergeSort(arr, l, m)
		mergeSort(arr, m+1, r)
		merge(arr, l, m, r)


arr = [12, 11, 13, 5, 6, 7]
n = len(arr)
print("Given array is")
for i in range(n):
	print("%d" % arr[i],end=" ")

mergeSort(arr, 0, n-1)
print("\n\nSorted array is")
for i in range(n):
	print("%d" % arr[i],end=" ")

#Median of Medians

def kthSmallest(array, l, r, k):
    if (k > 0 and k <= r - l + 1):
        n = r - l + 1
        medians = []
        i = 0
        while (i < n // 5):
            medians.append(Median(array, l + i * 5, 5))
            i += 1

        if (i * 5 < n):
            medians.append(Median(array, l + i * 5,n % 5))
            i += 1

        if i == 1:
            MedianofMedians = medians[i - 1]
        else:
            MedianofMedians = kthSmallest(medians, 0, i - 1, i // 2)

        position = partition(array, l, r, MedianofMedians)

        if (position - l == k - 1):
            return array[position]
        if (position - l > k - 1):

            return kthSmallest(array, l, position - 1, k)

        return kthSmallest(array, position + 1, r, k - position + l - 1)

    return 999999999999

def swap(array, a, b):
    temp = array[a]
    array[a] = array[b]
    array[b] = temp

def partition(array, l, r, x):
    for i in range(l, r):
        if array[i] == x:
            swap(array, r, i)
            break

    x = array[r]
    i = l
    for j in range(l, r):
        if (array[j] <= x):
            swap(array, i, j)
            i += 1
    swap(array, i, r)
    return i

def Median(arr, l, n):
    list = []
    for i in range(l, l + n):
        list.append(arr[i])

    list.sort()

    return list[n // 2]

L = list(map(int, input("").split()))

k = int(input(""))

print(kthSmallest(L, 0, len(L) - 1, k))

#Closet Pair

import math
class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

def dist(a, b):
	return math.sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y))

def bruteForce(P, n):
	min_dist = 99999999999
	for i in range(n):
		for j in range(i+1, n):
			if dist(P[i], P[j]) < min_dist:
				min_dist = dist(P[i], P[j])
	return min_dist

def min(x, y):
	if x < y:
		return x
	else:
		return y

def stripClosest(strip, size, f):
	min_dist = f
	strip = sorted(strip, key=lambda point: point.y)

	for i in range(size):
		for j in range(i+1, size):
			if (strip[j].y - strip[i].y) >= min_dist:
				break
			if dist(strip[i], strip[j]) < min_dist:
				min_dist = dist(strip[i], strip[j])
	return min_dist

def recur(P, n):
	if n <= 3:
		return bruteForce(P, n)
	mid = n//2
	mP = P[mid]
	left = recur(P, mid)
	right = recur(P[mid:], n - mid)
	d = min(left, right)
	strip = []
	for i in range(n):
		if abs(P[i].x - mP.x) < d:
			strip.append(P[i])
	return min(d, stripClosest(strip, len(strip), d))

def closestPair(P, n):
	P = sorted(P, key=lambda point: point.x)
	return recur(P, n)

P = []
n = int(input(""))
for i in range(n):
    coordinates = list(map(int, input("").split()))
    P.append(Point(x = coordinates[0],y = coordinates[1]))

print(closestPair(P, n))

#Integer Multiplication
binary_form_1 = str(input(""))
binary_form_2 = str(input(""))

def fitlen(str_1, str_2):
    if (len(str_1) < len(str_2)):
        str_1 = '0' * (len(str_2) - len(str_1)) + str_1
    else:
        str_2 = '0' * (len(str_1) - len(str_2)) + str_2
    return str_1, str_2

def add(str_1, str_2):
    if len(str_1) != len(str_2):
        str_1, str_2 = fitlen(str_1, str_2)

    res = ""
    carry = 0
    for i in range(len(str_1) - 1, -1, -1):
        a = int(str_1[i])
        b = int(str_2[i])
        val = a ^ b ^ carry
        res = str(val) + res
        carry = (a & b) | (a & carry) | (b & carry)
    if carry:
        res = '1' + res
    return res

def karatsuba(str_1, str_2):
   if len(str_1) != len(str_2):
       str_1, str_2 = fitlen(str_1, str_2)

   n = len(str_1)

   if n == 0:
       return 0
   if n == 1:
       return int(str_1[0])*int(str_2[0])

   first_half = n//2
   second_half = n - first_half

   left_1 = str_1[:first_half]
   right_1 = str_1[first_half:]

   left_2 = str_2[:first_half]
   right_2 = str_2[first_half:]

   res1 = karatsuba(left_1, left_2)
   res2 = karatsuba(right_1, right_2)
   res3 = karatsuba(add(left_1, right_1), add(left_2, right_2))

   return res1*(1<<(2*second_half)) + (res3 - res1 - res2) * (1 << second_half) + res2

print(karatsuba(binary_form_1, binary_form_2))

#Perfect Tree

import math

m = int(input(""))
leaf = list(map(int, input("").split()))
result = 0
level = int(math.log2(m))
level = 1 << level

def mergeSubTrees(list, low, mid, high):

    if list[mid] < list[mid + 1]:
        return False

    for i in range(low, mid + 1):
        temp = list[i]
        list[i] = list[i + (high - low + 1) // 2]
        list[i + (high - low + 1) // 2] = temp

    return True

def minOperations(list, low, high):
    global result
    if low < high:

        mid = (high + low) // 2

        minOperations(list, low, mid)

        minOperations(list, mid + 1, high)

        if mergeSubTrees(list, low, mid, high):
            result += 1

def is_sorted(leaf):
    for i in range(1, len(leaf)):
        if leaf[i] < leaf[i - 1]:
            return False

    return True

minOperations(leaf, 0, level - 1)

if is_sorted(leaf):
    print(result)
else:
    print(-1)

# HuffMan

string = str(input(""))
total = len(string.encode('utf-8')) * 8

class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return (self.left, self.right)

    def nodes(self):
        return (self.left, self.right)

    def __str__(self):
        return '%s_%s' % (self.left, self.right)


def huffman_code_tree(node, left = True, binString = ''):
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, True, binString + '0'))
    d.update(huffman_code_tree(r, False, binString + '1'))
    return d

freq = {}

for i in string:
    if i in freq:
        freq[i] += 1
    else:
        freq[i] = 1

freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

nodes = freq

while len(nodes) > 1:
    (key_1, c_1) = nodes[-1]
    (key_2, c_2) = nodes[-2]
    nodes = nodes[:-2]
    node = NodeTree(key_1, key_2)
    nodes.append((node, c_1 + c_2))

    nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

huffmanCode = huffman_code_tree(nodes[0][0])

saved_bit = 0

for (char, frequency) in freq:
    saved_bit += frequency*len(huffmanCode[char])

print(f'{saved_bit} {total - saved_bit}')


# FloodFill
n = int(input(""))

screen = []

for i in range(n):
    screen.append(list(map(int, input("").split())))

target_coordinate = list(map(int, input("").split()))
c = int(input(""))

def Recursion(screen, x, y, prev, new):
	if (x < 0 or x >= n or y < 0 or y >= n or screen[x][y] != prev or screen[x][y] == new):
		return

	screen[x][y] = new
	Recursion(screen, x + 1, y, prev, new)
	Recursion(screen, x - 1, y, prev, new)
	Recursion(screen, x, y + 1, prev, new)
	Recursion(screen, x, y - 1, prev, new)

def Fill(screen, x, y, new):
	prev = screen[x][y]
	if (prev != new):
	   Recursion(screen, x, y, prev, new)

Fill(screen, target_coordinate[0], target_coordinate[1], c)

for i in range(n):
	for j in range(n):
		print(screen[i][j], end=' ')
	print()


# Injected city

from collections import Counter

n = int(input(""))
graph = []

for i in range(n):
    graph.append(list(map(int, input("").split())))

initial = list(map(int, input("").split()))

sizes = [1 for _ in range(n)]
p = list(range(n))

def find(x):
    if p[x] != x:
        p[x] = find(p[x])
    return p[x]

def union(x, y):
    rx, ry = p[x], p[y]
    if rx < ry:
        p[ry] = rx
        sizes[rx] += sizes[ry]
    elif rx > ry:
        p[rx] = ry
        sizes[ry] += sizes[rx]

for i in range(n):
    for j in range(i + 1, n):
        if graph[i][j] == 1:
            union(i, j)

infected_by = Counter(find(i) for i in initial)

mx = -1
result = min(initial)

for i in initial:
    root = find(i)
    if infected_by[root] == 1:
         if sizes[root] > mx:
            mx, result = sizes[root], i
         elif sizes[root] == mx and i < result:
            mx, result = sizes[root], i

print(result)

# DeadLine scheduling

import heapq

n = int(input(""))
tasks_list = []
for i in range(n):
    tasks_list.append(list(map(int, input().strip().split())))

tasks_list = sorted(tasks_list, key=lambda elem: (elem[0], elem[1], elem[2]))
tasks_priority = []
current_time = 0
final_result = -999999999999

for i in tasks_list:
    while current_time < i[0] and len(tasks_priority) > 0:
        top_priority = heapq.heappop(tasks_priority)
        if i[0] - current_time >= top_priority[1]:
            current_time += top_priority[1]
            final_result = max(final_result, current_time - top_priority[0])
        else:
            top_priority[1] -= i[0] - current_time
            heapq.heappush(tasks_priority, top_priority)
            current_time = i[0]

    current_time = i[0]

    heapq.heappush(tasks_priority, i[1:])

while len(tasks_priority) > 0:
    top_priority = heapq.heappop(tasks_priority)
    final_result = max(
        final_result, current_time + top_priority[1] - top_priority[0]
    )
    current_time += top_priority[1]

print(final_result)
























