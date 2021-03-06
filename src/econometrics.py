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

            if you want to treat a varible as a categorical variable, add ".CAT" after the variable name.
            ex) y x_1 x_2.CAT x_3
            Add ".(base line value)" to set the base level.
            ex) y x_1 x_2.CAT.0 x_3
             

        Returns
        -------    
        result_df
        """
        self.data = data
        self.form = form
        self.build_form()
        self.estimate()
        self.coef_test(error=error, stata=True)
        
    def build_form(self):
        """
        Build form information dictionary.
        """
        self.form_idx = pd.DataFrame(False, index=self.data.columns, columns=["dependent", "independent"])
        self.variables = np.array([i.split(".")[0] for i in self.form.split(" ")])
        
        # categorical label
        self.cats = self.variables[np.where([(".CAT") in i for i in self.form.split(" ")])[0]]
        self.form_idx["categorical"] = False
        self.form_idx.loc[self.cats, "categorical"] = True
        self.form_idx["base_cat"] = np.nan
        for cat_i in self.cats:
            cat_i_info = np.array(self.form.split(" ")[int(np.where(self.variables==cat_i)[0])].split("."))
            print(cat_i_info)
            if len(cat_i_info) > 2: 
                num_base_cat = int(np.where(cat_i_info=="CAT")[0]) + 1
                self.form_idx.loc[cat_i, "base_cat"] = cat_i_info[num_base_cat]
            else:
                pass
        
        # dependent and independent label
        self.form_idx.loc[self.variables[0], "dependent"] = True
        for x in self.variables[1:]:
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

        cont_vars = self.data[self.form_idx[(self.form_idx["independent"])&(self.form_idx["categorical"] == False)].index]
        print(cont_vars)

        if self.form_idx["categorical"].sum() != 0:
            # cats = self.form_idx[self.form_idx["categorical"]].index
            for cat_i in self.cats:
                cat_dums = pd.get_dummies(self.data[cat_i], columns=cat_i)
                cat_dums.columns = [cat_i + "_" + str(i) for i in cat_dums.columns]
                print("NULL#?????", self.form_idx.loc[cat_i, "base_cat"])
                if np.isnan(np.float(self.form_idx.loc[cat_i, "base_cat"])):
                    # if no base category specified, drop first
                    base_cat = cat_dums.columns[0]
                else:
                    # drop specified base category
                    base_cat = cat_i + "_" + str(self.form_idx.loc[cat_i, "base_cat"])
                print("cat_dums", cat_dums)
                cat_dums = cat_dums.drop(base_cat, axis=1) #
            self.x = pd.concat([cont_vars, cat_dums], axis=1)

        else:
            self.x = cont_vars
        
        self.x.insert(0, "const", 1)
        self.x_names = self.x.columns
        self.x = self.x.values
        self.beta_hat = np.linalg.inv(self.x.T @ self.x) @ self.x.T @ self.y

    def coef_test(self, error="hetero", stata=False):
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
            if stata:
                # stata?????????????????????? --> ????????????
                # https://stackoverflow.com/questions/33155638/clustered-standard-errors-in-r-using-plm-with-fixed-effects
                c = (N / (N-K))
                self.vcme = c * self.vcme
                pass
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
    df = pd.read_csv("/mnt/d/0ngoing/my_projects/econometrics/src/BostonHousing.csv")

    reg = LinearRegression(df, form="medv crim rad.CAT.4", error="hetero")
    reg.result_df