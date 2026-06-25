def max_subarray_sum(arr, k):
    if len(arr) < k:
        return 0
    
    current_sum = sum(arr[:k])
    max_sum = current_sum
    
    for i in range(len(arr) - k):
        current_sum = current_sum - arr[i] + arr[i + k]
        if current_sum > max_sum:
            max_sum = current_sum
            
    return max_sum

def has_target_pair(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return True
        elif current_sum < target:
            left = left + 1
        else:
            right = right - 1
            
    return False

numbers = [1, 3, 2, 6, -1, 4, 1, 8, 2]
print(max_subarray_sum(numbers, 3))

sorted_numbers = [1, 2, 4, 4, 5, 6, 8, 9]
print(has_target_pair(sorted_numbers, 10))
