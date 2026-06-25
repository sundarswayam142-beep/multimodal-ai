def max_subarray_sum(arr, k):
    if len(arr) < k:
        return "Invalid array size"

    
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(len(arr) - k):
        window_sum = window_sum - arr[i] + arr[i + k]
        max_sum = max(max_sum, window_sum)
        
    return max_sum

def has_target_pair(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return True, (arr[left], arr[right])
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return False, None

test_nums = [2, 1, 5, 1, 3, 2]
print(f"Max subarray sum: {max_subarray_sum(test_nums, 3)}")

sorted_nums = [1, 2, 4, 6, 8, 9]
print(f"Contains target pair: {has_target_pair(sorted_nums, 10)}")
