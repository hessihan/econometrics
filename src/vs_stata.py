import numpy as np
import pandas as pd
import sys
sys.path.append("/mnt/d/0ngoing/my_projects/econometrics/src")
from econometrics import LinearRegression

if __name__ == "__main__":
    data_hetero = pd.read_csv("/mnt/d/0ngoing/my_projects/reg_from_scratch/src/data_hetero.csv", index_col=[0])
    reg_hetero = LinearRegression(data=data_hetero, form="y x_1 x_2 x_3", error="hetero")
    reg_hetero.result_df
    reg_homo = LinearRegression(data=data_hetero, form="y x_1 x_2 x_3", error="homo")
    reg_homo.result_df
    
    boston = pd.read_csv("/mnt/d/0ngoing/my_projects/reg_from_scratch/src/BostonHousing.csv")
    reg_homo = LinearRegression(boston, "medv crim age dis", error="homo")
    reg_homo.result_df

    reg_hetero = LinearRegression(boston, "medv crim age dis", error="hetero")
    reg_hetero.result_df