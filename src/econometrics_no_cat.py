# regression from scratch
# from os import error
# from numpy.testing._private.utils import build_err_msg
from os import error
import numpy as np
import pandas as pd
from scipy.stats import norm
# from scipy.stats import t

class LinearRegression():

    def __init__(self, data, form: str, error="hetero"):
        """
        Parameters
        ----------
        data: pd.DataFrame
        form: str
            variable names (form) of the model deliminated by space.
            ex) y x_1 x_2 x_3
        Returns
        -------    
        result_df
        """
        self.data = data
        self.form = form
        self.build_form()
        self.estimate()
        self.coef_test(error=error)
        
    def build_form(self):
        """
        Build form information dictionary.
        """
        self.form_idx = pd.DataFrame(False, index=self.data.columns, columns=["dependent", "independent"])
        self.form_idx.loc[self.form.split(" ")[0], "dependent"] = True
        for x in self.form.split(" ")[1:]:
            self.form_idx.loc[x, "independent"] = True

    def estimate(self):
        """
        Estimate (or fit) the parameter
        
        Parameters
        ----------
        Returns
        -------
        beta_hat
        """
        
        self.y = self.data[self.form_idx[self.form_idx["dependent"]].index].values
        self.x = self.data[self.form_idx[self.form_idx["independent"]].index]
        self.x.insert(0,"const",1)
        self.x_names = self.x.columns
        self.x = self.x.values
        self.beta_hat = np.linalg.inv(self.x.T @ self.x) @ self.x.T @ self.y

    def coef_test(self, error="hetero"):
        """
        Hypothesis Test for estimated coefficients
        
        Returns
        -------
        table
        """
        # residual
        y_hat = self.x @ self.beta_hat
        u_hat = self.y - y_hat
        # Wooldridge p-56.
        N = self.x.shape[0]
        K = self.x.shape[1]
        if error == "homo":
            SSR = np.sum(u_hat ** 2) 
            sigma_hat_sq = SSR / (N - K)
            # (the estimate of) variance-covariance-matrix-of-estimator: VCME
            self.vcme = sigma_hat_sq * np.linalg.inv(self.x.T @ self.x)
        elif error == "hetero":
            self.vcme = np.linalg.inv(self.x.T @ self.x) @ sum([u_hat[i, :] ** 2 * self.x[i, :].reshape(1, -1).T @ self.x[i, :].reshape(1, -1) for i in range(N)]) @ np.linalg.inv(self.x.T @ self.x)
            # self.vcme = np.dot(
            #     np.dot(np.linalg.inv(np.dot(self.x.T, self.x)), sum([u_hat[i, :] ** 2 * np.dot(self.x[i, :].reshape(1, -1).T, self.x[i, :].reshape(1, -1)) for i in range(N)])), 
            #     np.linalg.inv(np.dot(self.x.T, self.x)))
        self.std_err = np.sqrt(np.diag(self.vcme).reshape(-1, 1))
        self.t_stats = self.beta_hat / self.std_err
        self.p_value = 2 * (1 - norm.cdf(abs(self.t_stats)))
        self.ci_95_upper = self.beta_hat + norm.ppf(0.975) * self.std_err
        self.ci_95_lower = self.beta_hat - norm.ppf(0.975) * self.std_err
        # self.ci_95_upper = self.beta_hat + t.ppf(0.975, df=N-K) * self.std_err # stata use t dist for test, u ~ N() assumption needed
        # self.ci_95_lower = self.beta_hat - t.ppf(0.975, df=N-K) * self.std_err # stata use t dist for test, u ~ N() assumption needed

        self.result_df = pd.concat([
            pd.DataFrame(self.beta_hat), 
            pd.DataFrame(self.std_err), 
            pd.DataFrame(self.t_stats), 
            pd.DataFrame(self.p_value),
            pd.DataFrame(self.ci_95_lower),
            pd.DataFrame(self.ci_95_upper)
            ],axis=1)
        self.result_df.columns = ["coef", "std_err", "t_stats", "p_value", "CI_95_lower", "CI_95_upper"]
        self.result_df.index = self.x_names
        return self.result_df
        
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
    np.random.seed(0)
    x_1 = np.random.randn(sample_size)
    x_2 = 1 * np.random.randn(sample_size) + 3
    x_3 = 0.1 * np.random.randn(sample_size) + 3
    # sigma_list = np.random.randint(1, 5, size=sample_size)
    # sigma_list = np.random.randint(1, 5,sample_size)
    sigma_list = np.random.uniform(0.1, 10, sample_size)
    noise = np.random.randn(sample_size) * sigma_list
    y = 0.5 + (-9) * x_1 + 2 * x_2 + 2 * x_3 + noise
    data_hetero = pd.DataFrame(np.array([y, x_1, x_2, x_3]).T, columns=["y", "x_1", "x_2","x_3"])
    data_hetero.to_csv("/mnt/d/0ngoing/my_projects/reg_from_scratch/src/data_hetero.csv")

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
    np.diag(homo_vcme ) - np.diag(hetero_vcme)

    # t_stats, p_value
    reg.beta_hat
    reg.t_stats