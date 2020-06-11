import math

# Computes the mean and standard deviation of values in an array
def extract_mean_std(array):
    
    mean = sum(array) / len(array)
    std = math.sqrt(sum([pow(x - mean, 2) for x in array]) / len(array))

    return mean, std


