import copy
import numpy as np

# This class is for sorting a given array of integers.. The sorting algorithm is Merge Sort.
class Sorting:
    def sort_arr(self,unsorted_arr):
        self.arr = unsorted_arr
        self.size = len(unsorted_arr)
        self.merge_sort(0, self.size - 1)
        return self.arr

    # This method is used for merging
    def merge(self, l, m, r):
        left_most = m - l + 1
        right_most = r - m
        l_arr = copy.deepcopy(self.arr[l:l + left_most])
        r_arr = copy.deepcopy(self.arr[m+1:m+1+right_most])

        i = 0; j = 0; k = l
        while i < left_most and j < right_most :
            if l_arr[i] <= r_arr[j]:
                self.arr[k] = l_arr[i]
                i += 1
            else:
                self.arr[k] = r_arr[j]
                j += 1
            k += 1

        while i < left_most:
            self.arr[k] = l_arr[i]
            i += 1; k += 1

        while j < right_most:
            self.arr[k] = r_arr[j]
            j += 1; k += 1

    # This method is used for splitting the array into left and right sub-lists.
    def merge_sort(self, l, r):
        if l < r:
            m = (l+(r-1))//2
            self.merge_sort(l, m)
            self.merge_sort(m+1, r)
            self.merge(l, m, r)
        return

