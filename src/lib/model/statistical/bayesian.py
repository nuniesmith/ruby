import numpy as np
import pandas as pd

from lib.model._shims import logger

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False

try:
    from scipy import stats
    from scipy.optimize import minimize

    HAS_SCIPY = True
except ImportError:
    minimize = None
    stats = None
    HAS_SCIPY = False

try:
    import arviz as az  # For MCMC diagnostics and visualization

    HAS_ARVIZ = True
except ImportError:
    az = None  # type: ignore[assignment]
    HAS_ARVIZ = False

try:
    import pymc as pm  # For MCMC sampling

    HAS_PYMC = True
except ImportError:
    pm = None  # type: ignore[assignment]
    HAS_PYMC = False


class BayesianLinearRegression:
    """
    Bayesian Linear Regression model with options for both MLE and full Bayesian inference.
    """

    def __init__(
        self,
        alpha_prior_mean=178,
        alpha_prior_std=20,
        beta_prior_type="normal",
        beta_prior_mean=0,
        beta_prior_std=10,
        sigma_prior_max=50,
        asset=None,
        data_fetcher=None,
        **kwargs,
    ):
        """
        Initialize the Bayesian Linear Regression model with priors.

        Parameters:
        -----------
        alpha_prior_mean : float
            Mean of the prior for the intercept α
        alpha_prior_std : float
            Standard deviation of the prior for the intercept α
        beta_prior_type : str
            Type of prior for the slope β: 'normal' or 'lognormal'
        beta_prior_mean : float
            Mean of the prior for β (if normal) or log(β) (if lognormal)
        beta_prior_std : float
            Standard deviation of the prior for β (if normal) or log(β) (if lognormal)
        sigma_prior_max : float
            Upper bound for the uniform prior on σ
        asset : str, optional
            Name of the asset this model is for
        data_fetcher : object, optional
            Object for fetching data for the specified asset
        **kwargs : dict
            Additional keyword arguments
        """
        self.alpha_prior_mean = alpha_prior_mean
        self.alpha_prior_std = alpha_prior_std
        self.beta_prior_type = beta_prior_type
        self.beta_prior_mean = beta_prior_mean
        self.beta_prior_std = beta_prior_std
        self.sigma_prior_max = sigma_prior_max
        self.asset = asset
        self.data_fetcher = data_fetcher
        self.fitted_params = None
        self.xbar = None
        self.trace = None  # For storing MCMC samples
        self.model = None  # For storing PyMC model

    def plot_priors(self, x_range=None):
        """
        Plot the prior distributions for the model parameters.

        Parameters:
        -----------
        x_range : tuple, optional
            Range of x values to generate predictions from priors
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot prior for α
        x_alpha = np.linspace(
            self.alpha_prior_mean - 3 * self.alpha_prior_std, self.alpha_prior_mean + 3 * self.alpha_prior_std, 100
        )
        axes[0].plot(x_alpha, stats.norm.pdf(x_alpha, self.alpha_prior_mean, self.alpha_prior_std))
        axes[0].set_title(f"Prior for α: Normal({self.alpha_prior_mean}, {self.alpha_prior_std})")
        axes[0].set_xlabel("α (Intercept)")
        axes[0].set_ylabel("Density")

        # Plot prior for β
        if self.beta_prior_type == "normal":
            x_beta = np.linspace(
                self.beta_prior_mean - 3 * self.beta_prior_std, self.beta_prior_mean + 3 * self.beta_prior_std, 100
            )
            axes[1].plot(x_beta, stats.norm.pdf(x_beta, self.beta_prior_mean, self.beta_prior_std))
            axes[1].set_title(f"Prior for β: Normal({self.beta_prior_mean}, {self.beta_prior_std})")
        else:
            x_beta = np.linspace(0.01, np.exp(self.beta_prior_mean + 3 * self.beta_prior_std), 100)
            axes[1].plot(x_beta, stats.lognorm.pdf(x_beta, self.beta_prior_std, scale=np.exp(self.beta_prior_mean)))
            axes[1].set_title(f"Prior for β: LogNormal({self.beta_prior_mean}, {self.beta_prior_std})")
        axes[1].set_xlabel("β (Slope)")
        axes[1].set_ylabel("Density")

        # Plot prior for σ
        x_sigma = np.linspace(0, self.sigma_prior_max + 10, 100)
        axes[2].plot(x_sigma, stats.uniform.pdf(x_sigma, 0, self.sigma_prior_max))
        axes[2].set_title(f"Prior for σ: Uniform(0, {self.sigma_prior_max})")
        axes[2].set_xlabel("σ (Standard Deviation)")
        axes[2].set_ylabel("Density")

        plt.tight_layout()
        plt.show()

        # Plot prior predictive distributions if x_range is provided
        if x_range is not None:
            self._plot_prior_predictive(x_range)

    def _plot_prior_predictive(self, x_range):
        """
        Plot prior predictive distributions from the model's priors.

        Parameters:
        -----------
        x_range : tuple
            (min_x, max_x) for generating predictions
        """
        min_x, max_x = x_range
        x = np.linspace(min_x, max_x, 100)
        xbar = np.mean(x)

        # Simulate from priors
        n_lines = 100
        alpha = stats.norm.rvs(self.alpha_prior_mean, self.alpha_prior_std, size=n_lines)

        if self.beta_prior_type == "normal":
            beta = stats.norm.rvs(self.beta_prior_mean, self.beta_prior_std, size=n_lines)
        else:
            beta = stats.lognorm.rvs(self.beta_prior_std, scale=np.exp(self.beta_prior_mean), size=n_lines)

        sigma = stats.uniform.rvs(0, self.sigma_prior_max, size=n_lines)

        plt.figure(figsize=(10, 6))
        for i in range(n_lines):
            y = alpha[i] + beta[i] * (x - xbar)
            plt.plot(x, y, "k-", alpha=0.2)

        plt.title(f"Prior Predictive: {n_lines} lines from the prior")
        plt.xlabel("x")
        plt.ylabel("y")

        # Add reference lines for impossible values
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="y = 0")

        # Add prior predictive simulations with error
        plt.figure(figsize=(10, 6))
        for i in range(n_lines):
            mu = alpha[i] + beta[i] * (x - xbar)
            y = stats.norm.rvs(loc=mu, scale=sigma[i])
            plt.scatter(x, y, alpha=0.1, s=5, c="blue")

        plt.title(f"Prior Predictive: {n_lines} simulated datasets")
        plt.xlabel("x")
        plt.ylabel("y")

        plt.legend()
        plt.show()

    def negative_log_likelihood(self, params, x, y):
        """
        Calculate the negative log-likelihood for the linear model.

        Parameters:
        -----------
        params : list or array
            [α, β, σ] - Intercept, slope, and standard deviation
        x : array-like
            Predictor variable
        y : array-like
            Response variable

        Returns:
        --------
        float
            Negative log-likelihood value
        """
        alpha, beta, sigma = params
        mu = alpha + beta * (x - self.xbar)  # Centered predictor
        return -np.sum(stats.norm.logpdf(y, mu, sigma))

    def negative_log_posterior(self, params, x, y):
        """
        Calculate the negative log posterior (proportional) for MAP estimation.

        Parameters:
        -----------
        params : list or array
            [α, β, σ] - Intercept, slope, and standard deviation
        x : array-like
            Predictor variable
        y : array-like
            Response variable

        Returns:
        --------
        float
            Negative log posterior value
        """
        alpha, beta, sigma = params

        # Log-likelihood
        log_likelihood = np.sum(stats.norm.logpdf(y, alpha + beta * (x - self.xbar), sigma))

        # Log-prior for alpha
        log_prior_alpha = stats.norm.logpdf(alpha, self.alpha_prior_mean, self.alpha_prior_std)

        # Log-prior for beta
        if self.beta_prior_type == "normal":
            log_prior_beta = stats.norm.logpdf(beta, self.beta_prior_mean, self.beta_prior_std)
        else:
            log_prior_beta = stats.lognorm.logpdf(beta, self.beta_prior_std, scale=np.exp(self.beta_prior_mean))

        # Log-prior for sigma (uniform)
        log_prior_sigma = 0 if 0 < sigma <= self.sigma_prior_max else -np.inf

        # Negative log posterior
        return -(log_likelihood + log_prior_alpha + log_prior_beta + log_prior_sigma)

    def fit(self, x, y, method="mle", mcmc_samples=2000, tune=1000):
        """
        Fit the Bayesian linear regression model to data using either MLE or MCMC.

        Parameters:
        -----------
        x : array-like
            Predictor variable
        y : array-like
            Response variable
        method : str
            Fitting method: 'mle', 'map', or 'mcmc'
        mcmc_samples : int
            Number of MCMC samples if method is 'mcmc'
        tune : int
            Number of tuning steps for MCMC

        Returns:
        --------
        dict or arviz.InferenceData
            Fitted parameters and statistics (MLE/MAP) or MCMC trace
        """
        self.xbar = np.mean(x)
        x_centered = x - self.xbar

        if method.lower() == "mcmc":
            return self._fit_mcmc(x_centered, y, mcmc_samples, tune)
        else:
            return self._fit_optimize(x, y, method.lower())

    def _fit_optimize(self, x, y, method="mle"):
        """
        Fit the model using optimization (MLE or MAP).

        Parameters:
        -----------
        x : array-like
            Predictor variable
        y : array-like
            Response variable
        method : str
            'mle' or 'map'

        Returns:
        --------
        dict
            Fitted parameters and statistics
        """
        # Initial guesses based on priors
        initial_beta = self.beta_prior_mean if self.beta_prior_type == "normal" else np.exp(self.beta_prior_mean)

        initial_params = [self.alpha_prior_mean, initial_beta, self.sigma_prior_max / 2]

        # Define bounds for parameters
        beta_bounds = (None, None) if self.beta_prior_type == "normal" else (0.001, None)

        # Objective function to minimize
        if method == "mle":

            def objective_fn(params):
                return self.negative_log_likelihood(params, x, y)
        else:  # MAP

            def objective_fn(params):
                return self.negative_log_posterior(params, x, y)

        # Minimize negative log-likelihood/posterior
        result = minimize(  # type: ignore
            objective_fn,
            initial_params,
            bounds=[(None, None), beta_bounds, (0.001, None)],  # σ must be positive
            method="L-BFGS-B",
        )

        alpha_hat, beta_hat, sigma_hat = result.x
        self.fitted_params = {
            "alpha": alpha_hat,
            "beta": beta_hat,
            "sigma": sigma_hat,
            "xbar": self.xbar,
            "method": method,
        }

        # Calculate 89% credible intervals and covariance matrix
        if hasattr(result, "hess_inv"):
            hessian_inv = result.hess_inv if isinstance(result.hess_inv, np.ndarray) else result.hess_inv.todense()
            param_std = np.sqrt(np.diag(hessian_inv))

            ci_lower = [
                alpha_hat - 1.645 * param_std[0],
                beta_hat - 1.645 * param_std[1],
                sigma_hat - 1.645 * param_std[2],
            ]
            ci_upper = [
                alpha_hat + 1.645 * param_std[0],
                beta_hat + 1.645 * param_std[1],
                sigma_hat + 1.645 * param_std[2],
            ]

            self.fitted_params.update(
                {
                    "alpha_std": param_std[0],
                    "beta_std": param_std[1],
                    "sigma_std": param_std[2],
                    "alpha_ci": (ci_lower[0], ci_upper[0]),
                    "beta_ci": (ci_lower[1], ci_upper[1]),
                    "sigma_ci": (ci_lower[2], ci_upper[2]),
                    "cov_matrix": hessian_inv,
                }
            )

        return self.fitted_params

    def _fit_mcmc(self, x_centered, y, samples=2000, tune=1000):
        """
        Fit the model using MCMC sampling with PyMC.

        Parameters:
        -----------
        x_centered : array-like
            Centered predictor variable (x - xbar)
        y : array-like
            Response variable
        samples : int
            Number of MCMC samples
        tune : int
            Number of tuning steps

        Returns:
        --------
        arviz.InferenceData
            MCMC trace
        """
        with pm.Model() as model:
            # Priors
            alpha = pm.Normal("alpha", mu=self.alpha_prior_mean, sigma=self.alpha_prior_std)

            if self.beta_prior_type == "normal":
                beta = pm.Normal("beta", mu=self.beta_prior_mean, sigma=self.beta_prior_std)
            else:
                beta = pm.Lognormal("beta", mu=self.beta_prior_mean, sigma=self.beta_prior_std)

            sigma = pm.Uniform("sigma", lower=0, upper=self.sigma_prior_max)

            # Expected value of outcome
            mu = alpha + beta * x_centered

            # Likelihood (sampling distribution) of observations
            pm.Normal("y", mu=mu, sigma=sigma, observed=y)

            # Sample from the posterior
            trace = pm.sample(samples, tune=tune, return_inferencedata=True)

            # Calculate LOO and WAIC
            loo = az.loo(trace)
            waic = az.waic(trace)

            # Store the model and trace
            self.model = model
            self.trace = trace

            # Extract point estimates
            summary = az.summary(trace, hdi_prob=0.89)
            self.fitted_params = {
                "alpha": float(summary.loc["alpha", "mean"]),
                "beta": float(summary.loc["beta", "mean"]),
                "sigma": float(summary.loc["sigma", "mean"]),
                "alpha_ci": (float(summary.loc["alpha", "hdi_5.5%"]), float(summary.loc["alpha", "hdi_94.5%"])),
                "beta_ci": (float(summary.loc["beta", "hdi_5.5%"]), float(summary.loc["beta", "hdi_94.5%"])),
                "sigma_ci": (float(summary.loc["sigma", "hdi_5.5%"]), float(summary.loc["sigma", "hdi_94.5%"])),
                "loo": loo,
                "waic": waic,
                "xbar": self.xbar,
                "method": "mcmc",
            }

        return trace

    def plot_fit(self, x, y, prediction_interval=0.89, method="mean", n_samples=100):
        """
        Plot the data, fitted line, and prediction interval.

        Parameters:
        -----------
        x : array-like
            Predictor variable
        y : array-like
            Response variable
        prediction_interval : float
            Width of the prediction interval (between 0 and 1)
        method : str
            How to generate predictions: 'mean' or 'samples' (for MCMC only)
        n_samples : int
            Number of posterior samples to use for prediction bands
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before plotting results")

        alpha_hat = self.fitted_params["alpha"]
        beta_hat = self.fitted_params["beta"]
        sigma_hat = self.fitted_params["sigma"]
        xbar = self.fitted_params["xbar"]

        plt.figure(figsize=(10, 6))

        # Plot data points
        plt.scatter(x, y, alpha=0.6, color="blue", label="Data")

        # Generate prediction line
        x_seq = np.linspace(min(x), max(x), 100)
        x_centered = x_seq - xbar

        if self.trace is not None and method == "samples":
            # Use MCMC samples for predictions
            posterior_samples = self.trace.posterior

            # Plot posterior samples
            sample_idxs = np.random.choice(
                posterior_samples.chain.size * posterior_samples.draw.size, size=n_samples, replace=False
            )
            chain_idxs = sample_idxs // posterior_samples.draw.size
            draw_idxs = sample_idxs % posterior_samples.draw.size

            plt.figure(figsize=(10, 6))

            # Plot data
            plt.scatter(x, y, alpha=0.6, color="blue", label="Data")

            # Plot regression lines from posterior samples
            for i in range(n_samples):
                chain_idx = chain_idxs[i]
                draw_idx = draw_idxs[i]
                alpha_sample = float(posterior_samples.alpha[chain_idx, draw_idx].values)
                beta_sample = float(posterior_samples.beta[chain_idx, draw_idx].values)

                mu = alpha_sample + beta_sample * x_centered
                plt.plot(x_seq, mu, color="gray", alpha=0.1)

            # Plot mean prediction
            alpha_mean = float(posterior_samples.alpha.mean().values)
            beta_mean = float(posterior_samples.beta.mean().values)
            mu_mean = alpha_mean + beta_mean * x_centered
            plt.plot(x_seq, mu_mean, "r-", lw=2, label="Mean prediction")

            # Generate prediction intervals
            lower_percentile = 100 * (1 - prediction_interval) / 2
            upper_percentile = 100 - lower_percentile

            # Simulate predictions with uncertainty
            pred_samples = np.zeros((n_samples, len(x_seq)))

            for i in range(n_samples):
                chain_idx = chain_idxs[i]
                draw_idx = draw_idxs[i]
                alpha_sample = float(posterior_samples.alpha[chain_idx, draw_idx].values)
                beta_sample = float(posterior_samples.beta[chain_idx, draw_idx].values)
                sigma_sample = float(posterior_samples.sigma[chain_idx, draw_idx].values)

                mu = alpha_sample + beta_sample * x_centered
                pred_samples[i, :] = stats.norm.rvs(loc=mu, scale=sigma_sample)

            lower_bound = np.percentile(pred_samples, lower_percentile, axis=0)
            upper_bound = np.percentile(pred_samples, upper_percentile, axis=0)

            plt.fill_between(
                x_seq,
                lower_bound,
                upper_bound,
                color="gray",
                alpha=0.3,
                label=f"{int(prediction_interval * 100)}% Prediction Interval",
            )

        else:
            # Use point estimates for predictions
            mu_mean = alpha_hat + beta_hat * x_centered
            plt.plot(x_seq, mu_mean, "r-", lw=2, label="Mean prediction")

            # Generate prediction interval
            if prediction_interval > 0:
                lower_percentile = 100 * (1 - prediction_interval) / 2
                upper_percentile = 100 - lower_percentile

                # Simulate predictions
                n_sim = 1000
                sim_y = np.zeros((n_sim, len(x_seq)))
                for i in range(n_sim):
                    sim_y[i, :] = stats.norm.rvs(loc=mu_mean, scale=sigma_hat)

                lower_bound = np.percentile(sim_y, lower_percentile, axis=0)
                upper_bound = np.percentile(sim_y, upper_percentile, axis=0)

                plt.fill_between(
                    x_seq,
                    lower_bound,
                    upper_bound,
                    color="gray",
                    alpha=0.3,
                    label=f"{int(prediction_interval * 100)}% Prediction Interval",
                )

        plt.title(f"Linear Model: y = {alpha_hat:.2f} + {beta_hat:.2f} · (x - {xbar:.2f})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    def plot_posterior(self):
        """
        Plot the posterior distributions for MCMC fitted models.

        Returns:
        --------
        None
        """
        if self.trace is None:
            logger.warning("This method requires MCMC fitting. Use fit(method='mcmc')")
            return

        # Plot posterior distributions
        az.plot_posterior(self.trace, hdi_prob=0.89)
        plt.tight_layout()
        plt.show()

        # Plot trace
        az.plot_trace(self.trace)
        plt.tight_layout()
        plt.show()

    def plot_ppc(self):
        """
        Plot posterior predictive checks.

        Returns:
        --------
        None
        """
        if self.trace is None:
            logger.warning("This method requires MCMC fitting. Use fit(method='mcmc')")
            return

        # Plot posterior predictive checks
        if self.model is None:
            logger.warning("No PyMC model available for posterior predictive sampling.")
            return
        with self.model:
            pm.sample_posterior_predictive(self.trace, extend_inferencedata=True)

        az.plot_ppc(self.trace, kind="cumulative")
        plt.tight_layout()
        plt.show()

        az.plot_ppc(self.trace, kind="scatter")
        plt.tight_layout()
        plt.show()

    def summary(self):
        """
        Print a summary of the fitted model.

        Returns:
        --------
        None
        """
        if self.fitted_params is None:
            print("Model has not been fitted yet.")
            return

        alpha_hat = self.fitted_params["alpha"]
        beta_hat = self.fitted_params["beta"]
        sigma_hat = self.fitted_params["sigma"]
        method = self.fitted_params.get("method", "mle")

        print("=" * 60)
        print(f"Bayesian Linear Regression Summary (Method: {method.upper()})")
        print("=" * 60)
        print(f"Formula: y ~ N(α + β · (x - {self.xbar:.2f}), σ²)")
        print("-" * 60)
        print("Parameters:")
        print(f"α (Intercept): {alpha_hat:.4f}")

        if "alpha_ci" in self.fitted_params:
            alpha_ci = self.fitted_params["alpha_ci"]
            print(f"    89% CI: [{alpha_ci[0]:.4f}, {alpha_ci[1]:.4f}]")

        print(f"β (Slope): {beta_hat:.4f}")

        if "beta_ci" in self.fitted_params:
            beta_ci = self.fitted_params["beta_ci"]
            print(f"    89% CI: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}]")

        print(f"σ (Std Dev): {sigma_hat:.4f}")

        if "sigma_ci" in self.fitted_params:
            sigma_ci = self.fitted_params["sigma_ci"]
            print(f"    89% CI: [{sigma_ci[0]:.4f}, {sigma_ci[1]:.4f}]")

        print("-" * 60)

        if self.trace is not None:
            print("MCMC Info:")
            print(f"- Number of chains: {self.trace.posterior.chain.size}")
            print(f"- Samples per chain: {self.trace.posterior.draw.size}")
            if "loo" in self.fitted_params:
                print(f"- LOO: {self.fitted_params['loo'].loo:.2f}")
                print(f"- WAIC: {self.fitted_params['waic'].waic:.2f}")
            print("-" * 60)

        if "cov_matrix" in self.fitted_params:
            print("Covariance Matrix:")
            print(self.fitted_params["cov_matrix"])
            print("-" * 60)

        print("Model Interpretation:")
        print(f"- For each unit increase in x, y is expected to change by {beta_hat:.4f} units")
        print(f"- When x = {self.xbar:.2f}, the expected value of y is {alpha_hat:.4f}")
        print(f"- The standard deviation of the residuals is {sigma_hat:.4f}")
        print("=" * 60)

    def predict(self, x_new, return_std=False, samples=1000):
        """
        Make predictions for new x values.

        Parameters:
        -----------
        x_new : array-like
            New predictor values
        return_std : bool
            Whether to return standard deviations
        samples : int
            Number of posterior samples to use for MCMC predictions

        Returns:
        --------
        array or tuple
            Predicted means or (means, stds)
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before making predictions")

        alpha_hat = self.fitted_params["alpha"]
        beta_hat = self.fitted_params["beta"]
        sigma_hat = self.fitted_params["sigma"]
        xbar = self.fitted_params["xbar"]

        x_centered = np.asarray(x_new) - xbar

        if self.trace is not None:
            # Use MCMC samples
            posterior = self.trace.posterior

            # Get random samples
            idx = np.random.choice(posterior.chain.size * posterior.draw.size, size=samples, replace=True)
            chain_idx = idx // posterior.draw.size
            draw_idx = idx % posterior.draw.size

            # Get parameters from each sample
            alpha_samples = posterior.alpha.values[chain_idx, draw_idx]
            beta_samples = posterior.beta.values[chain_idx, draw_idx]

            # Calculate predictions for each sample and each x value
            preds = np.zeros((samples, len(x_centered)))

            for i in range(samples):
                preds[i, :] = alpha_samples[i] + beta_samples[i] * x_centered

            # Calculate mean and std of predictions
            pred_means = preds.mean(axis=0)
            pred_stds = preds.std(axis=0)
        else:
            # Use point estimates
            pred_means = alpha_hat + beta_hat * x_centered
            pred_stds = np.ones_like(x_centered) * sigma_hat

        if return_std:
            return pred_means, pred_stds
        else:
            return pred_means

    def get_waic(self):
        """
        Get the WAIC (Widely Applicable Information Criterion) for model comparison.
        Only available for MCMC models.

        Returns:
        --------
        float or None
            WAIC value or None if not available
        """
        if "waic" in self.fitted_params:
            return self.fitted_params["waic"]
        else:
            logger.warning("WAIC only available for MCMC fitted models")
            return None

    def get_loo(self):
        """
        Get the LOO (Leave-One-Out Cross-Validation) for model comparison.
        Only available for MCMC models.

        Returns:
        --------
        float or None
            LOO value or None if not available
        """
        if "loo" in self.fitted_params:
            return self.fitted_params["loo"]
        else:
            logger.warning("LOO only available for MCMC fitted models")
            return None

    @classmethod
    def compare_models(cls, models, criterion="waic"):
        """
        Compare multiple Bayesian models using WAIC or LOO.

        Parameters:
        -----------
        models : list
            List of BayesianLinearRegression instances
        criterion : str
            Model comparison criterion: 'waic' or 'loo'

        Returns:
        --------
        DataFrame
            Model comparison results
        """
        if criterion not in ("waic", "loo"):
            raise ValueError("Criterion must be 'waic' or 'loo'")

        results = []
        for i, model in enumerate(models):
            model_name = f"Model {i + 1}"
            waic = model.get_waic() if criterion == "waic" else model.get_loo()
            results.append({"Model": model_name, criterion.upper(): waic})

        return pd.DataFrame(results)
