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
