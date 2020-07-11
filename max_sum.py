# Python3 program to find maximum sum
# subarray in a given 2D array

# Implementation of Kadane's algorithm
# for 1D array. The function returns the
# maximum sum and stores starting and
# ending indexes of the maximum sum subarray
# at addresses pointed by start and finish
# pointers respectively.
def kadane(arr, start, finish, n):
    # initialize sum, maxSum and
    Sum = 0
    maxSum = -999999999999
    i = None

    # Just some initial value to check
    # for all negative values case
    finish[0] = -1

    # local variable
    local_start = 0

    for i in range(n):
        Sum += arr[i]
        if Sum < 0:
            Sum = 0
            local_start = i + 1
        elif Sum > maxSum:
            maxSum = Sum
            start[0] = local_start
            finish[0] = i

        # There is at-least one
    # non-negative number
    if finish[0] != -1:
        return maxSum

    # Special Case: When all numbers
    # in arr[] are negative
    maxSum = arr[0]
    start[0] = finish[0] = 0

    # Find the maximum element in array
    for i in range(1, n):
        if arr[i] > maxSum:
            maxSum = arr[i]
            start[0] = finish[0] = i
    return maxSum


# The main function that finds maximum
# sum rectangle in M[][]
def findMaxSum(M):
    global ROW, COL

    # Variables to store the final output
    maxSum, finalLeft = -999999999999, None
    finalRight, finalTop, finalBottom = None, None, None
    left, right, i = None, None, None

    temp = [None] * ROW
    Sum = 0
    start = [0]
    finish = [0]

    # Set the left column
    for left in range(COL):

        # Initialize all elements of temp as 0
        temp = [0] * ROW

        # Set the right column for the left
        # column set by outer loop
        for right in range(left, COL):

            # Calculate sum between current left
            # and right for every row 'i'
            for i in range(ROW):
                temp[i] += M[i][right]

            # Find the maximum sum subarray in
            # temp[]. The kadane() function also
            # sets values of start and finish.
            # So 'sum' is sum of rectangle between
            # (start, left) and (finish, right) which
            # is the maximum sum with boundary columns
            # strictly as left and right.
            Sum = kadane(temp, start, finish, ROW)

            # Compare sum with maximum sum so far.
            # If sum is more, then update maxSum
            # and other output values
            if Sum > maxSum:
                maxSum = Sum
                finalLeft = left
                finalRight = right
                finalTop = start[0]
                finalBottom = finish[0]

            # Prfinal values
    print("(Top, Left)", "(", finalTop,
          finalLeft, ")")
    print("(Bottom, Right)", "(", finalBottom,
          finalRight, ")")
    print("Max sum is:", maxSum)


# Driver Code
ROW = 4
COL = 5
M = [[1, 2, -1, -4, -20],
     [-8, -3, 4, 2, 1],
     [3, 8, 10, 1, 3],
     [-4, -1, 1, 7, -6]]

findMaxSum(M)

# This code is contributed by PranchalK
