# regression from scratch
import numpy as np
import pandas as pd

class LinearRegression():

    def __init__(self) -> None:
        self.form = None
        pass

    def estimate(x, y):
        """
        Estimate (or fit) the parameter
        
        Parameters
        ----------
        x : array_like
            Explanatory variables.
        y : array_like
            Explained variable.
        """
        x = np.insert(x, 0, 1, axis=1)
        return np.linalg.inv(x.T @ x) @ x.T @ y

# Debug
if __name__ == "__main__":
    print("hello")
    print(np.array([0, 1, 2]))
