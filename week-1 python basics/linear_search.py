def locate_target_reading(readings_list, minimum_allowed):
    matching_indices = []
    
    for idx in range(len(readings_list)):
        if readings_list[idx] >= minimum_allowed:
            matching_indices.append(idx)
            
    return matching_indices

def check_value_exists(items, target_item):
    for item in items:
        if item == target_item:
            return True
    return False

voltages = [12, 45, 98, 23, 104, 56, 89]
print(locate_target_reading(voltages, 80))

parts_inventory = ["casing", "bolt", "bracket", "valve"]
print(check_value_exists(parts_inventory, "valve"))
