import numpy as np
import unittest
from src.utils_n2oblife.useful.Sorting import (
    bubble_sort, insertion_sort, merge_sort, quicksort, timsort
)

class TestSortingAlgorithms(unittest.TestCase):
    def setUp(self):
        """Set up test cases with different types of input arrays."""
        self.test_cases = [
            [],  # Empty list
            [1],  # Single element
            [2, 1],  # Two elements
            [5, 3, 8, 6, 2, 7, 4, 1],  # Unsorted list
            [1, 2, 3, 4, 5, 6, 7, 8],  # Already sorted
            [8, 7, 6, 5, 4, 3, 2, 1],  # Reverse sorted
            [5, 1, 3, 2, 5, 3, 1, 2],  # List with duplicates
            list(np.random.randint(0, 1000, 50)),  # Large random list
        ]

    def _run_sort_test(self, sort_function):
        """Helper function to run a test on all test cases for a given sorting function."""
        for array in self.test_cases:
            with self.subTest(array=array):
                sorted_array = sort_function(array.copy())  # Sort a copy to avoid mutation
                self.assertEqual(sorted_array, sorted(array))  # Compare with Python's sorted()

    def test_bubble_sort(self):
        self._run_sort_test(bubble_sort)

    def test_insertion_sort(self):
        self._run_sort_test(insertion_sort)

    def test_merge_sort(self):
        self._run_sort_test(merge_sort)

    def test_quicksort(self):
        self._run_sort_test(quicksort)

    def test_timsort(self):
        self._run_sort_test(timsort)

