import numpy as np

def bubble_sort(array:list)->list:
    """A O(n^2) algorithm to sort array

    Args:
        array (list): An array to sort

    Returns:
        list: A sorted array
    """
    n = len(array)
    for i in range(n):
        # terminate early if there's nothing left to sort
        already_sorted = True
        for j in range(n - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
                already_sorted = False
        if already_sorted:
            break
    return array

def insertion_sort(array:list)->list:
    """A O(n^2) algorithm to sort array

    Args:
        array (list): An array to sort

    Returns:
        list: A sorted array
    """
    for i in range(1, len(array)):
        key_item = array[i]
        j = i - 1
        while j >= 0 and array[j] > key_item:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = key_item
    return array

def merge(left:list, right:list)->list:
    """Function used for merge sort

    Args:
        left (list): An array to sort
        right (list): An array to sort

    Returns:
        list: A sorted array
    """
    if len(left) == 0:
        return right
    if len(right) == 0:
        return left
    result = []
    index_left = index_right = 0

    while len(result) < len(left) + len(right):
        # we use a cursor on the arrays while comparing the elements to build new array 
        if left[index_left] <= right[index_right]:
            result.append(left[index_left])
            index_left += 1
        else:
            result.append(right[index_right])
            index_right += 1
        # at the end of the array just add the rest of the list
        if index_right == len(right):
            result += left[index_left:]
            break
        if index_left == len(left):
            result += right[index_right:]
            break
    return result

def merge_sort(array:list)->list:
    """A O(n*log2(n)) algorithm to sort array

    Args:
        array (list): An array to sort

    Returns:
        list: A sorted array
    """
    if len(array) < 2:
        return array
    # with random midpoint better complexity in general
    midpoint = np.random.randint(0, len(array)-1)
    return merge(
        left=merge_sort(array[:midpoint]),
        right=merge_sort(array[midpoint:]))

def quicksort(array:list)->list:
    """A O(n*log2(n)) algorithm to sort array

    Args:
        array (list): An array to sort

    Returns:
        list: A sorted array
    """
    if len(array) < 2:
        return array
    
    low, same, high = [], [], []
    # Select your `pivot` element randomly
    pivot = array[np.random.randint(0, len(array) - 1)]

    for item in array:
        if item < pivot:
            low.append(item)
        elif item == pivot:
            same.append(item)
        elif item > pivot:
            high.append(item)
    return quicksort(low) + same + quicksort(high)

def modified_insertion_sort(array:list, left=0, right:int=None)->list:
    """Used only for the timsort algorithm"""
    if right is None:
        right = len(array) - 1

    for i in range(left + 1, right + 1):
        key_item = array[i]
        j = i - 1
        while j >= left and array[j] > key_item:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = key_item
    return array

def timsort(array:list, min_run = 32)->list:
    """A O(n*log2(n)) algorithm to sort array

    Args:
        array (list): An array to sort

    Returns:
        list: A sorted array
    """
    n = len(array)
    # Start by slicing and sorting small portions of the array
    for i in range(0, n, min_run):
        modified_insertion_sort(array, i, min((i + min_run - 1), n - 1))

    # Merge the sorted slices.
    # Start from `min_run`, doubling the size until surpassing array's length
    size = min_run
    while size < n:
        for start in range(0, n, size * 2):
            midpoint = start + size - 1
            end = min((start + size * 2 - 1), (n-1))
            merged_array = merge(
                left=array[start:midpoint + 1],
                right=array[midpoint + 1:end + 1])
            array[start:start + len(merged_array)] = merged_array
        size *= 2
    return array