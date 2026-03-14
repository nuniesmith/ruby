import numpy as np

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


class GaussianModel:
    """
    Bayesian Gaussian model for a single variable.
    """

    def __init__(
        self,
        mu_prior_mean=178,
        mu_prior_std=20,
        sigma_prior_alpha=2,
        sigma_prior_beta=0.1,
        asset=None,
        data_fetcher=None,
        **kwargs,
    ):
        """
        Initialize the Gaussian model with prior beliefs about parameters.

        Parameters:
        -----------
        mu_prior_mean : float
            Mean of the prior distribution for the mean parameter μ
        mu_prior_std : float
            Standard deviation of the prior distribution for the mean parameter μ
        sigma_prior_alpha : float
            Shape parameter for the inverse-gamma prior on σ²
        sigma_prior_beta : float
            Scale parameter for the inverse-gamma prior on σ²
        asset : str
            Asset identifier (e.g., 'bitcoin', 'gold')
        data_fetcher : object
            Data fetcher instance for retrieving asset data
        **kwargs : dict
            Additional parameters that will be ignored (for compatibility with config)
        """
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_std = mu_prior_std
        self.sigma_prior_alpha = sigma_prior_alpha
        self.sigma_prior_beta = sigma_prior_beta
        self.posterior_samples = None
        self.fitted_params = None

        # Store asset information
        self.asset = asset
        self.data_fetcher = data_fetcher

        # Log unexpected parameters for debugging
        if kwargs:
            unexpected_params = list(kwargs.keys())
            if len(unexpected_params) > 0:
                logger.debug(f"GaussianModel ignoring unexpected parameters: {unexpected_params}")

    def plot_priors(self, num_samples=1000):
        """
        Plot the prior distributions for μ and σ with samples.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot prior for μ
        x_mu = np.linspace(self.mu_prior_mean - 3 * self.mu_prior_std, self.mu_prior_mean + 3 * self.mu_prior_std, 100)
        ax1.plot(x_mu, stats.norm.pdf(x_mu, self.mu_prior_mean, self.mu_prior_std))

        # Show samples from prior
        mu_samples = np.random.normal(self.mu_prior_mean, self.mu_prior_std, num_samples)
        ax1.hist(mu_samples, bins=30, density=True, alpha=0.3, color="blue")

        ax1.set_title(f"Prior for μ: Normal({self.mu_prior_mean}, {self.mu_prior_std})")
        ax1.set_xlabel("μ")
        ax1.set_ylabel("Density")

        # Plot prior for σ (using inverse-gamma prior for σ²)
        sigma_squared_samples = stats.invgamma.rvs(
            self.sigma_prior_alpha, scale=self.sigma_prior_beta, size=num_samples
        )
        sigma_samples = np.sqrt(sigma_squared_samples)

        ax2.hist(sigma_samples, bins=30, density=True, alpha=0.3, color="red")

        x_sigma = np.linspace(0, np.percentile(sigma_samples, 95) * 1.5, 100)
        sigma_squared_densities = stats.invgamma.pdf(x_sigma**2, self.sigma_prior_alpha, scale=self.sigma_prior_beta)
        # Chain rule for transformation of density
        sigma_densities = 2 * x_sigma * sigma_squared_densities
        ax2.plot(x_sigma, sigma_densities)

        ax2.set_title(f"Prior for σ: σ² ~ Inv-Gamma({self.sigma_prior_alpha}, {self.sigma_prior_beta})")
        ax2.set_xlabel("σ")
        ax2.set_ylabel("Density")

        plt.tight_layout()
        plt.show()

    def log_prior(self, params):
        """
        Calculate the log prior probability for the given parameters.

        Parameters:
        -----------
        params : list or array
            [μ, σ] - Mean and standard deviation of the Gaussian

        Returns:
        --------
        float
            Log prior probability
        """
        mu, sigma = params

        # Prior for μ is Normal(mu_prior_mean, mu_prior_std)
        log_prior_mu = stats.norm.logpdf(mu, self.mu_prior_mean, self.mu_prior_std)

        # Prior for σ² is Inverse-Gamma(sigma_prior_alpha, sigma_prior_beta)
        log_prior_sigma_squared = stats.invgamma.logpdf(sigma**2, self.sigma_prior_alpha, scale=self.sigma_prior_beta)
        # Chain rule adjustment for transformation from σ² to σ
        log_prior_sigma = log_prior_sigma_squared + np.log(2) + np.log(sigma)

        return log_prior_mu + log_prior_sigma

    def negative_log_likelihood(self, params, data):
        """
        Calculate the negative log-likelihood for the Gaussian model.

        Parameters:
        -----------
        params : list or array
            [μ, σ] - Mean and standard deviation of the Gaussian
        data : array-like
            Observed data

        Returns:
        --------
        float
            Negative log-likelihood value
        """
        mu, sigma = params
        return -np.sum(stats.norm.logpdf(data, mu, sigma))

    def negative_log_posterior(self, params, data):
        """
        Calculate the negative log posterior probability (proportional to).

        Parameters:
        -----------
        params : list or array
            [μ, σ] - Mean and standard deviation of the Gaussian
        data : array-like
            Observed data

        Returns:
        --------
        float
            Negative log posterior probability
        """
        log_prior = self.log_prior(params)
        log_likelihood = -self.negative_log_likelihood(params, data)

        # Return negative log posterior (for minimization)
        return -(log_prior + log_likelihood)

    def fit(self, data, method="MAP"):
        """
        Fit the Gaussian model to data using maximum a posteriori (MAP) estimation
        or MCMC sampling.

        Parameters:
        -----------
        data : array-like
            Observed data
        method : str
            'MAP' for maximum a posteriori or 'MCMC' for Markov Chain Monte Carlo

        Returns:
        --------
        dict
            Fitted parameters and statistics
        """
        data = np.asarray(data)

        if method == "MAP":
            # Initial guesses based on priors
            initial_params = [self.mu_prior_mean, np.sqrt(self.sigma_prior_beta / (self.sigma_prior_alpha - 1))]

            # Minimize negative log posterior
            result = minimize(  # type: ignore[operator, reportOptionalCall]
                lambda params: self.negative_log_posterior(params, data),
                initial_params,
                bounds=[(None, None), (0.001, None)],  # σ must be positive
                method="L-BFGS-B",
            )

            mu_hat, sigma_hat = result.x
            self.fitted_params = {"mu": mu_hat, "sigma": sigma_hat, "method": "MAP"}

            # Calculate approximate 89% credible intervals (5.5% to 94.5%)
            if hasattr(result, "hess_inv"):
                try:
                    hessian_inv = (
                        result.hess_inv if isinstance(result.hess_inv, np.ndarray) else result.hess_inv.todense()
                    )
                    param_std = np.sqrt(np.diag(hessian_inv))

                    ci_lower = [mu_hat - 1.645 * param_std[0], sigma_hat - 1.645 * param_std[1]]
                    ci_upper = [mu_hat + 1.645 * param_std[0], sigma_hat + 1.645 * param_std[1]]

                    # Ensure σ is positive in CI
                    ci_lower[1] = max(0.001, ci_lower[1])

                    self.fitted_params.update(
                        {
                            "mu_std": param_std[0],
                            "sigma_std": param_std[1],
                            "mu_ci": (ci_lower[0], ci_upper[0]),
                            "sigma_ci": (ci_lower[1], ci_upper[1]),
                        }
                    )
                except Exception:
                    # Fallback if Hessian calculation fails
                    self.fitted_params.update({"mu_std": None, "sigma_std": None, "mu_ci": None, "sigma_ci": None})

            return self.fitted_params

        elif method == "MCMC":
            try:
                # If scipy.stats has a MCMCSampler, use it; otherwise, we'll implement a simple MCMC
                import emcee  # Requires emcee package
                from scipy.stats import invgamma, norm

                # Define log probability function for emcee
                def log_probability(theta, data):
                    mu, log_sigma = theta
                    sigma = np.exp(log_sigma)  # Work with log(σ) for better sampling

                    # Prior
                    log_prior_mu = norm.logpdf(mu, self.mu_prior_mean, self.mu_prior_std)
                    log_prior_sigma_squared = invgamma.logpdf(
                        sigma**2, self.sigma_prior_alpha, scale=self.sigma_prior_beta
                    )
                    log_prior_sigma = log_prior_sigma_squared + np.log(2) + np.log(sigma)

                    # Check for invalid values
                    if not np.isfinite(log_prior_mu) or not np.isfinite(log_prior_sigma) or sigma <= 0:
                        return -np.inf

                    # Likelihood
                    log_likelihood = np.sum(norm.logpdf(data, mu, sigma))

                    return log_prior_mu + log_prior_sigma + log_likelihood

                # Initial guess and setup
                initial_mu = np.mean(data)
                initial_sigma = np.log(np.std(data))

                ndim = 2  # Number of parameters
                nwalkers = 32  # Number of MCMC walkers

                # Add small random noise to initial positions
                pos = np.array([initial_mu, initial_sigma]) + 1e-4 * np.random.randn(nwalkers, ndim)

                # Setup sampler
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(data,))

                # Run MCMC
                print("Running MCMC...")
                sampler.run_mcmc(pos, 5000, progress=True)

                # Discard burn-in and flatten chain
                samples = sampler.get_chain(discard=1000, flat=True)

                # Convert log(σ) back to σ
                samples_transformed = np.column_stack([samples[:, 0], np.exp(samples[:, 1])])
                self.posterior_samples = samples_transformed

                # Calculate summary statistics
                mu_samples = samples_transformed[:, 0]
                sigma_samples = samples_transformed[:, 1]

                mu_mean = np.mean(mu_samples)
                mu_std = np.std(mu_samples)
                sigma_mean = np.mean(sigma_samples)
                sigma_std = np.std(sigma_samples)

                # Calculate 89% credible intervals
                mu_ci = np.percentile(mu_samples, [5.5, 94.5])
                sigma_ci = np.percentile(sigma_samples, [5.5, 94.5])

                self.fitted_params = {
                    "mu": mu_mean,
                    "sigma": sigma_mean,
                    "mu_std": mu_std,
                    "sigma_std": sigma_std,
                    "mu_ci": tuple(mu_ci),
                    "sigma_ci": tuple(sigma_ci),
                    "method": "MCMC",
                }

                return self.fitted_params

            except ImportError:
                print("MCMC requires emcee package. Falling back to MAP estimation.")
                return self.fit(data, method="MAP")
        else:
            raise ValueError("Method must be either 'MAP' or 'MCMC'")

    def plot_posterior(self, bins=30):
        """
        Plot the posterior distributions of μ and σ.

        Parameters:
        -----------
        bins : int
            Number of bins for histograms
        """
        if self.posterior_samples is None:
            if self.fitted_params is None:
                raise ValueError("Model must be fitted before plotting posterior")
            print("Warning: No posterior samples available. Showing MAP estimate only.")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            mu_hat = self.fitted_params["mu"]
            sigma_hat = self.fitted_params["sigma"]

            # If we have confidence intervals, use them
            if "mu_ci" in self.fitted_params and self.fitted_params["mu_ci"] is not None:
                mu_low, mu_high = self.fitted_params["mu_ci"]
                sigma_low, sigma_high = self.fitted_params["sigma_ci"]

                # Create dummy data for illustration
                x_mu = np.linspace(mu_low - (mu_high - mu_low), mu_high + (mu_high - mu_low), 100)
                mu_std = (mu_high - mu_low) / (2 * 1.645)
                ax1.plot(x_mu, stats.norm.pdf(x_mu, mu_hat, mu_std))

                x_sigma = np.linspace(
                    max(0, sigma_low - (sigma_high - sigma_low)), sigma_high + (sigma_high - sigma_low), 100
                )
                sigma_std = (sigma_high - sigma_low) / (2 * 1.645)
                ax2.plot(x_sigma, stats.norm.pdf(x_sigma, sigma_hat, sigma_std))
            else:
                # Just mark the MAP estimates
                ax1.axvline(mu_hat, color="blue")
                ax2.axvline(sigma_hat, color="red")

            ax1.set_title(f"Posterior for μ (MAP estimate: {mu_hat:.2f})")
            ax1.set_xlabel("μ")
            ax1.set_ylabel("Density")

            ax2.set_title(f"Posterior for σ (MAP estimate: {sigma_hat:.2f})")
            ax2.set_xlabel("σ")
            ax2.set_ylabel("Density")

        else:
            # Plot full posterior from MCMC samples
            mu_samples = self.posterior_samples[:, 0]
            sigma_samples = self.posterior_samples[:, 1]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot posterior for μ
            ax1.hist(mu_samples, bins=bins, density=True, alpha=0.6, color="blue")
            ax1.axvline(np.mean(mu_samples), color="black", linestyle="-", label="Mean")
            ax1.axvline(np.percentile(mu_samples, 5.5), color="black", linestyle="--", alpha=0.5, label="89% CI")
            ax1.axvline(np.percentile(mu_samples, 94.5), color="black", linestyle="--", alpha=0.5)

            ax1.set_title(f"Posterior for μ (Mean: {np.mean(mu_samples):.2f})")
            ax1.set_xlabel("μ")
            ax1.set_ylabel("Density")
            ax1.legend()

            # Plot posterior for σ
            ax2.hist(sigma_samples, bins=bins, density=True, alpha=0.6, color="red")
            ax2.axvline(np.mean(sigma_samples), color="black", linestyle="-", label="Mean")
            ax2.axvline(np.percentile(sigma_samples, 5.5), color="black", linestyle="--", alpha=0.5, label="89% CI")
            ax2.axvline(np.percentile(sigma_samples, 94.5), color="black", linestyle="--", alpha=0.5)

            ax2.set_title(f"Posterior for σ (Mean: {np.mean(sigma_samples):.2f})")
            ax2.set_xlabel("σ")
            ax2.set_ylabel("Density")
            ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_fit(self, data, show_samples=5):
        """
        Plot the data histogram and fitted Gaussian distribution.

        Parameters:
        -----------
        data : array-like
            Observed data
        show_samples : int
            Number of posterior samples to show (if MCMC was used)
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before plotting results")

        mu_hat = self.fitted_params["mu"]
        sigma_hat = self.fitted_params["sigma"]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram of data
        ax.hist(data, bins=20, density=True, alpha=0.6, color="grey", edgecolor="black", label="Data")

        # Plot range
        x = np.linspace(min(data) - sigma_hat, max(data) + sigma_hat, 200)

        if self.posterior_samples is not None and show_samples > 0:
            # Plot a few random samples from the posterior
            indices = np.random.choice(len(self.posterior_samples), show_samples, replace=False)
            for i in indices:
                mu_i, sigma_i = self.posterior_samples[i]
                y = stats.norm.pdf(x, mu_i, sigma_i)
                ax.plot(x, y, "b-", lw=0.5, alpha=0.3)

            # Plot the average model
            y_mean = stats.norm.pdf(x, mu_hat, sigma_hat)
            ax.plot(x, y_mean, "r-", lw=2, label="Fitted Gaussian (Posterior Mean)")
        else:
            # Plot the MAP/MLE model
            y = stats.norm.pdf(x, mu_hat, sigma_hat)
            ax.plot(x, y, "r-", lw=2, label="Fitted Gaussian")

        # Add vertical line at μ
        ax.axvline(x=mu_hat, color="r", linestyle="--", alpha=0.5)
        ax.text(mu_hat, 0.02, f"μ = {mu_hat:.2f}", rotation=90, verticalalignment="bottom")

        # Add title with credible intervals if available
        if "mu_ci" in self.fitted_params and self.fitted_params["mu_ci"] is not None:
            mu_ci = self.fitted_params["mu_ci"]
            sigma_ci = self.fitted_params["sigma_ci"]
            title = (
                f"Fitted Gaussian: μ = {mu_hat:.2f} [{mu_ci[0]:.2f}, {mu_ci[1]:.2f}], "
                f"σ = {sigma_hat:.2f} [{sigma_ci[0]:.2f}, {sigma_ci[1]:.2f}]"
            )
        else:
            title = f"Fitted Gaussian: μ = {mu_hat:.2f}, σ = {sigma_hat:.2f}"

        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        plt.show()

    def predict(self, n_samples=1000, method="exact"):
        """
        Generate predictive samples from the fitted model.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        method : str
            'exact' for exact predictive or 'sample' to use posterior samples

        Returns:
        --------
        array
            Samples from the predictive distribution
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before making predictions")

        if method == "exact" or self.posterior_samples is None:
            # Use fitted parameters directly
            mu = self.fitted_params["mu"]
            sigma = self.fitted_params["sigma"]
            return np.random.normal(mu, sigma, n_samples)
        else:
            # Use posterior samples for fully Bayesian prediction
            # Randomly select posterior samples
            indices = np.random.choice(len(self.posterior_samples), n_samples, replace=True)
            samples = self.posterior_samples[indices]

            # For each selected posterior sample, generate one predictive sample
            predictions = np.array([np.random.normal(mu_i, sigma_i) for mu_i, sigma_i in samples])

            return predictions

    def plot_predictive(self, n_samples=1000, method="exact"):
        """
        Plot the predictive distribution.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        method : str
            'exact' for exact predictive or 'sample' to use posterior samples
        """
        samples = self.predict(n_samples, method)

        plt.figure(figsize=(10, 6))
        plt.hist(samples, bins=30, density=True, alpha=0.7)

        # Add density curve
        x = np.linspace(min(samples), max(samples), 200)
        if method == "exact" or self.posterior_samples is None:
            # For exact method, we can compute the exact predictive density
            mu = self.fitted_params["mu"]
            sigma = self.fitted_params["sigma"]
            y = stats.norm.pdf(x, mu, sigma)
            plt.plot(x, y, "r-", lw=2, label="Predictive Density")
        else:
            # For sample method, use KDE to estimate the density
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(samples)
            y = kde(x)
            plt.plot(x, y, "r-", lw=2, label="Predictive Density (KDE)")

        plt.title("Predictive Distribution")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

        return samples
