import math
from itertools import islice

def monte_carlo_integrate(integrand, sampler, measure=1.0, n=100):
    # Sum elements and elements squared
    total = 0.0
    total_sq = 0.0
    for x in islice(sampler, n):
        f = integrand(x)
        total += f
        total_sq += (f**2)
    # Return answer
    sample_mean = total/n
    sample_var = (total_sq - (total**2)/n)/(n-1.0)
    return (measure*sample_mean, measure*math.sqrt(sample_var/n))
