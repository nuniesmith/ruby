import pickle

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

try:
    from sklearn.model_selection import KFold

    HAS_SKLEARN = True
except ImportError:
    KFold = None
    HAS_SKLEARN = False


class PolynomialRegression:
    """
    Polynomial Regression model using a Bayesian approach.
    """

    def __init__(
        self,
        degree=2,
        alpha_prior_mean=0,
        alpha_prior_std=10,
        beta_prior_mean=0,
        beta_prior_std=5,
        sigma_prior_max=10,
        regularization=0,
        optimizer="BFGS",
    ):
        """
        Initialize the Polynomial Regression model with priors.

        Parameters:
        -----------
        degree : int
            Degree of the polynomial
        alpha_prior_mean : float
            Mean of the prior for the intercept α
        alpha_prior_std : float
            Standard deviation of the prior for the intercept α
        beta_prior_mean : float
            Mean of the prior for all β coefficients
        beta_prior_std : float
            Standard deviation of the prior for all β coefficients
        sigma_prior_max : float
            Upper bound for the uniform prior on σ
        regularization : float
            Regularization strength for L2 regularization (ridge)
        optimizer : str
            Optimization method to use ('BFGS', 'Powell', 'Nelder-Mead', etc.)
        """
        self.degree = degree
        self.alpha_prior_mean = alpha_prior_mean
        self.alpha_prior_std = alpha_prior_std
        self.beta_prior_mean = beta_prior_mean
        self.beta_prior_std = beta_prior_std
        self.sigma_prior_max = sigma_prior_max
        self.regularization = regularization
        self.optimizer = optimizer
        self.fitted_params = None
        self.x_means = None

    def _create_polynomial_features(self, x):
        """
        Create polynomial features from the input data.

        Parameters:
        -----------
        x : array-like
            Predictor variable

        Returns:
        --------
        array
            Matrix of polynomial features
        """
        x = np.asarray(x).reshape(-1)
        X = np.zeros((len(x), self.degree + 1))

        for i in range(self.degree + 1):
            X[:, i] = x**i

        return X

    def _center_features(self, X):
        """
        Center the features matrix for better numerical stability.

        Parameters:
        -----------
        X : array
            Matrix of polynomial features

        Returns:
        --------
        array
            Centered matrix of polynomial features
        """
        X_centered = X.copy()
        self.x_means = np.mean(X, axis=0)

        # Don't center the intercept column
        for i in range(1, X.shape[1]):
            X_centered[:, i] = X[:, i] - self.x_means[i]

        return X_centered

    def negative_log_likelihood(self, params, X, y):
        """
        Calculate the negative log-likelihood for the polynomial model.

        Parameters:
        -----------
        params : list or array
            [α, β₁, β₂, ..., βₙ, σ] - Intercept, coefficients, and standard deviation
        X : array
            Matrix of polynomial features
        y : array-like
            Response variable

        Returns:
        --------
        float
            Negative log-likelihood value
        """
        sigma = params[-1]
        coefs = params[:-1]

        # Linear predictor
        mu = X @ coefs

        # Basic negative log-likelihood
        nll = -np.sum(stats.norm.logpdf(y, mu, sigma))

        # Add L2 regularization (ridge) if specified
        if self.regularization > 0:
            # Don't regularize intercept (coefs[0])
            ridge_penalty = self.regularization * np.sum(coefs[1:] ** 2)
            nll += ridge_penalty

        return nll

    def fit(self, x, y):
        """
        Fit the polynomial regression model to data using MLE.

        Parameters:
        -----------
        x : array-like
            Predictor variable
        y : array-like
            Response variable

        Returns:
        --------
        dict
            Fitted parameters and statistics
        """
        # Input validation
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        if len(x) != len(y):
            raise ValueError("Input arrays x and y must have the same length")

        if len(x) <= self.degree:
            raise ValueError(f"Not enough data points ({len(x)}) for polynomial of degree {self.degree}")

        # Create and center polynomial features
        X = self._create_polynomial_features(x)
        X_centered = self._center_features(X)

        # Initial guesses based on priors
        initial_alpha = self.alpha_prior_mean
        initial_betas = [self.beta_prior_mean] * self.degree
        initial_sigma = self.sigma_prior_max / 2

        initial_params = [initial_alpha] + initial_betas + [initial_sigma]

        # Bounds for parameters (only σ is constrained to be positive)
        param_bounds = [(None, None)] * (self.degree + 1) + [(0.001, None)]

        # Minimize negative log-likelihood
        result = minimize(  # type: ignore[operator, reportOptionalCall]
            lambda params: self.negative_log_likelihood(params, X_centered, y),
            initial_params,
            bounds=param_bounds,
            method=self.optimizer,
        )

        fitted_coefs = result.x[:-1]
        sigma_hat = result.x[-1]

        # Store results
        self.fitted_params = {"coefficients": fitted_coefs, "sigma": sigma_hat, "x_means": self.x_means}

        # Calculate AIC and BIC
        n_params = len(fitted_coefs) + 1  # coefficients + sigma
        n_samples = len(y)
        log_lik = -self.negative_log_likelihood(result.x, X_centered, y)

        aic = -2 * log_lik + 2 * n_params
        bic = -2 * log_lik + n_params * np.log(n_samples)

        self.fitted_params.update({"aic": aic, "bic": bic, "log_likelihood": log_lik})

        # Calculate 89% credible intervals and covariance matrix
        if hasattr(result, "hess_inv"):
            try:
                # Try to get the Hessian inverse
                hessian_inv = result.hess_inv if isinstance(result.hess_inv, np.ndarray) else result.hess_inv.todense()

                # Add small ridge to ensure positive definiteness
                hessian_inv = hessian_inv + np.eye(hessian_inv.shape[0]) * 1e-8

                param_std = np.sqrt(np.diag(hessian_inv))

                ci_lower = result.x - 1.645 * param_std
                ci_upper = result.x + 1.645 * param_std

                self.fitted_params.update(
                    {"param_std": param_std, "ci_lower": ci_lower, "ci_upper": ci_upper, "cov_matrix": hessian_inv}
                )

            except Exception as e:
                logger.warning(f"Could not compute uncertainty estimates: {e}")

        return self.fitted_params

    def predict(self, x_new, return_std=False, return_interval=False, interval_width=0.89):
        """
        Make predictions for new data points.

        Parameters:
        -----------
        x_new : array-like
            New predictor values
        return_std : bool
            Whether to return the standard deviation
        return_interval : bool
            Whether to return the prediction interval
        interval_width : float
            Width of the prediction interval (between 0 and 1)

        Returns:
        --------
        tuple or array
            (mean predictions, [std], [lower_bound, upper_bound]) depending on parameters
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before making predictions")

        # Create polynomial features
        X_new = self._create_polynomial_features(x_new)

        # Center the features
        X_new_centered = X_new.copy()
        for i in range(1, X_new.shape[1]):
            X_new_centered[:, i] = X_new[:, i] - self.fitted_params["x_means"][i]

        # Calculate mean predictions
        mean_pred = X_new_centered @ self.fitted_params["coefficients"]

        # Calculate prediction intervals if needed
        if not return_std and not return_interval:
            return mean_pred

        sigma = self.fitted_params["sigma"]

        if return_interval:
            z_score = stats.norm.ppf((1 + interval_width) / 2)

            # Analytic prediction intervals
            lower_bound = mean_pred - z_score * sigma
            upper_bound = mean_pred + z_score * sigma

            if return_std:
                return mean_pred, sigma, (lower_bound, upper_bound)
            else:
                return mean_pred, (lower_bound, upper_bound)

        # Return only mean and std
        return mean_pred, sigma

    def score(self, x, y):
        """
        Calculate R-squared value for the model.

        Parameters:
        -----------
        x : array-like
            Predictor variable
        y : array-like
            Response variable

        Returns:
        --------
        float
            R-squared score
        """
        y_pred = self.predict(x)
        y = np.asarray(y).flatten()
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)

        # R-squared calculation
        r_squared = 1 - (ss_residual / ss_total)

        # Adjusted R-squared
        n = len(y)
        p = self.degree + 1  # Number of parameters
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

        return {"r_squared": r_squared, "adj_r_squared": adj_r_squared, "rmse": np.sqrt(ss_residual / n)}

    def cross_validate(self, x, y, k=5, scoring="r_squared"):
        """
        Perform k-fold cross-validation.

        Parameters:
        -----------
        x : array-like
            Predictor variable
        y : array-like
            Response variable
        k : int
            Number of folds
        scoring : str
            Scoring metric to use ('r_squared', 'adj_r_squared', 'rmse')

        Returns:
        --------
        tuple
            (mean score, std score)
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        kf = KFold(n_splits=k, shuffle=True, random_state=42)  # type: ignore[reportOptionalCall]
        scores = []

        for train_idx, test_idx in kf.split(x):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.fit(x_train, y_train)
            score_dict = self.score(x_test, y_test)

            if scoring == "rmse":
                # For RMSE, lower is better
                scores.append(score_dict["rmse"])
            else:
                # For R-squared and adjusted R-squared, higher is better
                scores.append(score_dict[scoring])

        return np.mean(scores), np.std(scores)

    @staticmethod
    def select_best_degree(x, y, max_degree=10, min_degree=1, criterion="aic", k_folds=5, regularization=0):
        """
        Find the optimal polynomial degree based on AIC, BIC, or cross-validation.

        Parameters:
        -----------
        x : array-like
            Predictor variable
        y : array-like
            Response variable
        max_degree : int
            Maximum polynomial degree to consider
        min_degree : int
            Minimum polynomial degree to consider
        criterion : str
            Criterion to use ('aic', 'bic', 'cv_r_squared', 'cv_rmse')
        k_folds : int
            Number of folds for cross-validation
        regularization : float
            Regularization strength

        Returns:
        --------
        int
            Best polynomial degree
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        results = []
        cv_based = criterion.startswith("cv_")

        for degree in range(min_degree, max_degree + 1):
            model = PolynomialRegression(degree=degree, regularization=regularization)

            try:
                if cv_based:
                    # Cross-validation based criteria
                    scoring = criterion[3:]  # Remove 'cv_' prefix
                    mean_score, _ = model.cross_validate(x, y, k=k_folds, scoring=scoring)

                    # For RMSE, lower is better; for R-squared, higher is better
                    score = -mean_score if scoring == "rmse" else mean_score
                    results.append((degree, score))
                else:
                    # Information criteria based
                    model.fit(x, y)
                    results.append((degree, model.fitted_params[criterion.lower()]))
            except Exception as e:
                logger.warning(f"Error for degree {degree}: {e}")
                # Skip this degree
                continue

        if not results:
            raise ValueError("Could not compute scores for any polynomial degrees")

        # For AIC/BIC/RMSE: lower is better, for R-squared: higher is better
        if cv_based and criterion != "cv_rmse":
            best_degree = max(results, key=lambda x: x[1])[0]
        else:
            best_degree = min(results, key=lambda x: x[1])[0]

        return best_degree

    def plot_fit(
        self, x, y, prediction_interval=0.89, num_points=100, plot_residuals=False, plot_qq=False, figsize=(12, 8)
    ):
        """
        Plot the data, fitted curve, and prediction interval.

        Parameters:
        -----------
        x : array-like
            Predictor variable
        y : array-like
            Response variable
        prediction_interval : float
            Width of the prediction interval (between 0 and 1)
        num_points : int
            Number of points to use for plotting the curve
        plot_residuals : bool
            Whether to plot residuals
        plot_qq : bool
            Whether to plot a QQ plot for residuals
        figsize : tuple
            Figure size
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before plotting results")

        # Convert inputs to numpy arrays
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        # Determine number of subplots
        n_plots = 1 + int(plot_residuals) + int(plot_qq)
        fig = plt.figure(figsize=figsize)

        # Main plot with data and fitted curve
        ax1 = fig.add_subplot(n_plots, 1, 1)

        # Plot data points
        ax1.scatter(x, y, alpha=0.6, color="blue", label="Data")

        # Generate prediction curve
        x_seq = np.linspace(min(x), max(x), num_points)
        lower_bound: np.ndarray | None = None
        upper_bound: np.ndarray | None = None
        if prediction_interval:
            result = self.predict(x_seq, return_interval=True, interval_width=prediction_interval)
            y_mean, (lower_bound, upper_bound) = result  # type: ignore[misc]
        else:
            y_mean = self.predict(x_seq)

        ax1.plot(x_seq, y_mean, "r-", lw=2, label="Mean prediction")

        # Generate prediction interval
        if prediction_interval > 0 and lower_bound is not None and upper_bound is not None:
            ax1.fill_between(
                x_seq,
                lower_bound,
                upper_bound,
                color="gray",
                alpha=0.3,
                label=f"{int(prediction_interval * 100)}% Prediction Interval",
            )

        ax1.set_title(f"Polynomial Regression (degree {self.degree})")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()

        # Add residual plot if requested
        if plot_residuals:
            ax2 = fig.add_subplot(n_plots, 1, 2)
            y_pred = self.predict(x)
            residuals = y - y_pred

            ax2.scatter(y_pred, residuals, alpha=0.6, color="green")
            ax2.axhline(y=0, color="r", linestyle="-")
            ax2.set_title("Residuals vs Fitted Values")
            ax2.set_xlabel("Fitted Values")
            ax2.set_ylabel("Residuals")

        # Add QQ plot if requested
        if plot_qq:
            ax3 = fig.add_subplot(n_plots, 1, n_plots)
            y_pred = self.predict(x)
            residuals = y - y_pred

            # Standardize residuals
            std_residuals = residuals / np.std(residuals)

            # Create QQ plot
            stats.probplot(std_residuals, dist="norm", plot=ax3)
            ax3.set_title("QQ Plot of Standardized Residuals")

        plt.tight_layout()
        plt.show()

    def summary(self):
        """
        Print a summary of the fitted polynomial model.

        Returns:
        --------
        None
        """
        if self.fitted_params is None:
            print("Model has not been fitted yet.")
            return

        print("=" * 60)
        print(f"Polynomial Regression Summary (degree {self.degree})")
        print("=" * 60)

        coefs = self.fitted_params["coefficients"]
        sigma = self.fitted_params["sigma"]

        print("Formula:")
        formula = "y ~ N("
        formula += f"{coefs[0]:.4f}"

        for i in range(1, len(coefs)):
            if coefs[i] >= 0:
                formula += f" + {coefs[i]:.4f} · (x^{i} - {self.fitted_params['x_means'][i]:.4f})"
            else:
                formula += f" - {-coefs[i]:.4f} · (x^{i} - {self.fitted_params['x_means'][i]:.4f})"

        formula += f", {sigma:.4f}²)"
        print(formula)
        print("-" * 60)

        print("Parameters:")
        has_ci = "ci_lower" in self.fitted_params and "ci_upper" in self.fitted_params
        has_std = "param_std" in self.fitted_params

        for i, coef in enumerate(coefs):
            param_name = "α (Intercept)" if i == 0 else f"β_{i} (x^{i})"

            if has_ci and has_std:
                ci_lower = self.fitted_params["ci_lower"][i]
                ci_upper = self.fitted_params["ci_upper"][i]
                std_err = self.fitted_params["param_std"][i]
                z_value = coef / std_err
                p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))

                print(f"{param_name}: {coef:.4f} ± {std_err:.4f} [89% CI: ({ci_lower:.4f}, {ci_upper:.4f})]")
                print(f"  z-value: {z_value:.4f}, p-value: {p_value:.4f}")
            else:
                print(f"{param_name}: {coef:.4f}")

        print(f"σ (Residual std): {sigma:.4f}")
        print("-" * 60)

        # Print model fit statistics
        if "log_likelihood" in self.fitted_params:
            print(f"Log-likelihood: {self.fitted_params['log_likelihood']:.4f}")
        if "aic" in self.fitted_params:
            print(f"AIC: {self.fitted_params['aic']:.4f}")
        if "bic" in self.fitted_params:
            print(f"BIC: {self.fitted_params['bic']:.4f}")

        print("=" * 60)

    def save_model(self, filename):
        """
        Save model parameters to file.

        Parameters:
        -----------
        filename : str
            Path to save the model file
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "fitted_params": self.fitted_params,
            "degree": self.degree,
            "alpha_prior_mean": self.alpha_prior_mean,
            "alpha_prior_std": self.alpha_prior_std,
            "beta_prior_mean": self.beta_prior_mean,
            "beta_prior_std": self.beta_prior_std,
            "sigma_prior_max": self.sigma_prior_max,
            "regularization": self.regularization,
            "optimizer": self.optimizer,
        }

        with open(filename, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filename}")

    @classmethod
    def load_model(cls, filename):
        """
        Load model from file.

        Parameters:
        -----------
        filename : str
            Path to the model file

        Returns:
        --------
        PolynomialRegression
            Loaded model
        """
        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        # Create a new model instance with the same hyperparameters
        model = cls(
            degree=model_data["degree"],
            alpha_prior_mean=model_data["alpha_prior_mean"],
            alpha_prior_std=model_data["alpha_prior_std"],
            beta_prior_mean=model_data["beta_prior_mean"],
            beta_prior_std=model_data["beta_prior_std"],
            sigma_prior_max=model_data["sigma_prior_max"],
            regularization=model_data["regularization"],
            optimizer=model_data["optimizer"],
        )

        # Load the fitted parameters
        model.fitted_params = model_data["fitted_params"]

        logger.info(f"Model loaded from {filename}")
        return model

    def get_residuals(self, x, y):
        """
        Calculate residuals for diagnostic purposes.

        Parameters:
        -----------
        x : array-like
            Predictor variable
        y : array-like
            Response variable

        Returns:
        --------
        dict
            Dictionary with different types of residuals and statistics
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before calculating residuals")

        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        # Get predictions
        y_pred = self.predict(x)

        # Calculate raw residuals
        raw_residuals = y - y_pred

        # Calculate standardized residuals
        std_residuals = raw_residuals / self.fitted_params["sigma"]

        # Calculate studentized residuals
        # This requires leverage values, which we approximate
        X = self._create_polynomial_features(x)
        X_centered = X.copy()
        for i in range(1, X.shape[1]):
            X_centered[:, i] = X[:, i] - self.fitted_params["x_means"][i]

        # Hat matrix diagonal (leverage values)
        try:
            H = X_centered @ np.linalg.inv(X_centered.T @ X_centered) @ X_centered.T
            leverage = np.diag(H)

            # Studentized residuals
            student_residuals = raw_residuals / (self.fitted_params["sigma"] * np.sqrt(1 - leverage))
        except Exception:
            # If matrix inversion fails, use standardized residuals
            leverage = np.zeros_like(x)
            student_residuals = std_residuals

        return {
            "raw": raw_residuals,
            "standardized": std_residuals,
            "studentized": student_residuals,
            "leverage": leverage,
        }
