
import numpy as np
import pandas as pd
import sys
sys.path.append("/mnt/d/0ngoing/my_projects/reg_from_scratch/src")
from econometrics import LinearRegression

# Debug
if __name__ == "__main__":
    np.random.seed(100)
    sample_size = 10000

    # Moc data: homo is true
    x_1 = np.random.randn(sample_size)
    x_2 = 1 * np.random.randn(sample_size) + 10
    x_3 = 0.1 * np.random.randn(sample_size) + 3
    noise = 10 * np.random.randn(sample_size)
    y = 0.5 + (-0.8) * x_1 + -0.2 * x_2 + 0.2 * x_3 + noise
    data = pd.DataFrame(np.array([y, x_1, x_2, x_3]).T, columns=["y", "x_1", "x_2","x_3"])

    # test
    reg = LinearRegression(data=data, form="y x_1 x_2 x_3")
    reg.coef_test(error="homo")
    homo_vcme = reg.vcme
    reg.coef_test(error="hetero")
    hetero_vcme = reg.vcme
    abs(hetero_vcme) < abs(homo_vcme)
    true_vcme_homo = 1 * np.linalg.inv(reg.x.T @ reg.x)

    # vcme estimates accuracy
    (true_vcme_homo - homo_vcme) / true_vcme_homo
    (true_vcme_homo - hetero_vcme) / true_vcme_homo

    # Moc data: hetero is true
    np.random.seed(12094)
    x_1 = np.random.randn(sample_size)
    x_2 = 10 * np.random.randn(sample_size) + 10
    x_3 = 0.1 * np.random.randn(sample_size) + 3
    # sigma_list = np.random.randint(1, 5, size=sample_size)
    sigma_list = np.random.randn(sample_size) ** 2
    noise = np.random.randn(sample_size) * sigma_list
    y = 300 + (-9) * x_1 + 2 * x_2 + 20 * x_3 + noise
    data_hetero = pd.DataFrame(np.array([y, x_1, x_2, x_3]).T, columns=["y", "x_1", "x_2","x_3"])

    N = reg.x.shape[0]
    true_vcme_hetero = np.linalg.inv(reg.x.T @ reg.x) @ sum([sigma_list[i] ** 2 * reg.x[i, :].reshape(1, -1).T @ reg.x[i, :].reshape(1, -1) for i in range(N)]) @ np.linalg.inv(reg.x.T @ reg.x)

    reg = LinearRegression(data=data_hetero, form="y x_1 x_2 x_3")
    reg.coef_test(error="homo")
    homo_vcme = reg.vcme
    reg.coef_test(error="hetero")
    hetero_vcme = reg.vcme

    # vcme estimates accuracy
    err_rate_homo = (true_vcme_hetero - homo_vcme) / true_vcme_hetero
    err_rate_hetero = (true_vcme_hetero - hetero_vcme) / true_vcme_hetero
    abs(err_rate_homo) > abs(err_rate_hetero)
    np.diag(homo_vcme) - np.diag(hetero_vcme)

    # t_stats, p_value
    print("beta_hat")
    reg.beta_hat
    print("t_stats")
    reg.t_stats
    print("p_value")
    reg.p_value