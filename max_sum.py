def get_bounds(arr):
    first = 0
    last = 0
    for i in range(len(arr)):
        if arr[i] < 3:
            last += 1
        else:
            first = i
            last = i
    return first + 1, last
