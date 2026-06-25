class BatchScanner:
    def __init__(self, batch_id, readings):
        
        self.batch_id = batch_id
        self.readings = readings

    def get_failed_count(self, threshold):
        fail_count = 0
        for val in self.readings:
            if val < threshold:
                fail_count = fail_count + 1
        return fail_count

    def check_stability(self):
        for i in range(len(self.readings) - 1):
            difference = abs(self.readings[i] - self.readings[i+1])
            if difference > 50:
                return "Unstable"
        return "Stable"

sensor_data = [92, 88, 45, 90, 32, 95]
inspector = BatchScanner("BATCH-404", sensor_data)

print(inspector.get_failed_count(85))
print(inspector.check_stability())
