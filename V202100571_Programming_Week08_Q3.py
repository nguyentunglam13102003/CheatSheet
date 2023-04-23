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



