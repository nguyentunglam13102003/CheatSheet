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


