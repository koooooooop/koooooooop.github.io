#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def example_function(x):
    """
    A simple example function for demonstration
    f(x) = x^2 + 2x + 1
    """
    return x**2 + 2*x + 1

# Main script
if __name__ == "__main__":
    # Generate data
    x = np.linspace(-10, 10, 100)
    y = example_function(x)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.grid(True)
    plt.title("Example Function: f(x) = x^2 + 2x + 1")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    
    # Print some sample values
    sample_x = np.array([-5, -2, 0, 2, 5])
    sample_y = example_function(sample_x)
    
    print("Sample Values:")
    for i, j in zip(sample_x, sample_y):
        print(f"f({i}) = {j}") 