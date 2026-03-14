import pickle
from typing import Any

import numpy as np

from lib.model._shims import ModelEvaluator, log_execution, logger

try:
    from sklearn.metrics import f1_score, precision_score, recall_score

    HAS_SKLEARN = True
except ImportError:
    precision_score = None
    recall_score = None
    f1_score = None
    HAS_SKLEARN = False

try:
    from hmmlearn import hmm

    HAS_HMMLEARN = True
except ImportError:
    hmm = None
    HAS_HMMLEARN = False

from lib.model.base.model import BaseModel


class HMMModel(BaseModel):
    """
    A class to encapsulate ARIMA (SARIMAX) functionality.
    Now implemented as a subclass of BaseModel to ensure a consistent API.
    """

    def __init__(self, n_components=4, cov_type="diag", n_iter=1000, tol=1e-4, verbose=True):
        """
        Initialize the Hidden Markov Model with parameters.
        """
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.best_model = None
        self.best_log_likelihood = -np.inf
        self.metrics = {}
        self._initialize_model()

    def _initialize_model(self):
        """Initialize a new Gaussian HMM model instance."""
        if not HAS_HMMLEARN:
            raise ImportError("hmmlearn is required for HMMModel")
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.cov_type,
            n_iter=self.n_iter,
            tol=self.tol,
            verbose=self.verbose,
        )
        # Initialize startprob_ to ensure it sums to 1
        self.model.startprob_ = np.full(self.n_components, 1 / self.n_components)

    @log_execution
    def build_model(self, **kwargs) -> Any:
        """Build the HMM model."""
        self._initialize_model()
        return self.model

    @log_execution
    def fit(self, X_train, **kwargs) -> Any:
        """Fit the HMM model to the training data."""
        self.model.fit(X_train)
        return self.model

    @log_execution
    def train(self, X_train, early_stopping_tol=1e-4, patience=5):
        """
        Train the HMM model using early stopping based on log-likelihood improvement.

        Args:
            X_train: Training data (e.g., NumPy array).
            early_stopping_tol: Minimum improvement in log-likelihood to reset patience.
            patience: Number of iterations to wait for improvement.

        Returns:
            tuple: The best parameter configuration (if applicable) and best log-likelihood.
        """
        logger.info("Starting HMM model training...")
        no_improvement_count = 0

        for iteration in range(self.n_iter):
            try:
                self.model.fit(X_train)
                log_likelihood = self.model.score(X_train)
                logger.info(f"Iteration {iteration + 1}, Log-Likelihood: {log_likelihood}")

                # Update best model if log-likelihood improves
                if log_likelihood > self.best_log_likelihood + early_stopping_tol:
                    self.best_log_likelihood = log_likelihood
                    # Deep copy the model using pickle
                    self.best_model = pickle.loads(pickle.dumps(self.model))
                    no_improvement_count = 0  # reset patience counter
                    logger.info("New best model found and saved.")
                else:
                    no_improvement_count += 1

                # Early stopping based on patience
                if no_improvement_count >= patience:
                    logger.info(f"Early stopping at iteration {iteration + 1} due to lack of improvement.")
                    break

            except Exception as e:
                logger.error(f"Error during HMM model training at iteration {iteration + 1}: {e}")
                raise

        # Use the best model if available
        self.model = self.best_model if self.best_model is not None else self.model
        logger.info("HMM model training completed.")
        return self.best_log_likelihood

    @log_execution
    def predict(self, X):
        """
        Predict hidden states using the trained HMM model.

        Args:
            X: Data to predict on.

        Returns:
            np.ndarray: Predicted hidden states.
        """
        if self.model is None:
            raise ValueError("HMMModel is not trained. Call train() first.")

        try:
            logger.info(f"Predicting hidden states for data with shape {X.shape}...")
            predictions = self.model.predict(X)
            logger.info(f"Prediction completed with {len(predictions)} hidden states.")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    @log_execution
    def evaluate(self, X_test, y_true, y_pred):
        """
        Evaluate the trained HMM model using log-likelihood, AIC, BIC, and classification metrics.

        Args:
            X_test: Test data.
            y_true: True labels for classification evaluation.
            y_pred: Predicted labels for classification evaluation.

        Returns:
            dict: Evaluation metrics.
        """
        try:
            logger.info("Evaluating HMM model...")
            log_likelihood = self.model.score(X_test)

            # Calculate number of parameters
            n_params = (
                self.model.n_components**2  # transition matrix
                + self.model.n_components * self.model.n_features * 2  # means and covariances
            )
            n_samples = X_test.shape[0]

            # Compute AIC and BIC
            aic = 2 * n_params - 2 * log_likelihood
            bic = np.log(n_samples) * n_params - 2 * log_likelihood

            # Evaluate classification metrics
            if HAS_SKLEARN:
                precision = precision_score(y_true, y_pred, average="weighted")  # type: ignore[operator]
                recall = recall_score(y_true, y_pred, average="weighted")  # type: ignore[operator]
                f1 = f1_score(y_true, y_pred, average="weighted")  # type: ignore[operator]
            else:
                precision = None
                recall = None
                f1 = None

            classification_metrics = {}
            if ModelEvaluator is not None:
                evaluator = ModelEvaluator(y_true, y_pred)
                classification_metrics = evaluator.calculate_metrics()

            logger.info(f"Evaluation Results - Log-Likelihood: {log_likelihood}, AIC: {aic}, BIC: {bic}")
            logger.info(f"Classification Metrics: {classification_metrics}")

            return {
                "log_likelihood": log_likelihood,
                "AIC": aic,
                "BIC": bic,
                "classification_metrics": classification_metrics,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        except Exception as e:
            logger.error(f"Error evaluating HMM model: {e}")
            raise

    @log_execution
    def save_model(self, file_path):
        """Save the trained HMM model to a file."""
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self.best_model or self.model, f)
            logger.info(f"Model saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving HMM model: {e}")
            raise

    @log_execution
    def load_model(self, file_path):
        """Load the HMM model from a file."""
        try:
            with open(file_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading HMM model: {e}")
            raise

    @log_execution
    def generate_samples(self, n_samples=10):
        """
        Generate samples from the trained HMM model.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            tuple: (samples, hidden_states)
        """
        try:
            logger.info(f"Generating {n_samples} samples from the HMM model...")
            samples, hidden_states = self.model.sample(n_samples)
            logger.info(f"Generated {n_samples} samples. Hidden states: {hidden_states}")
            return samples, hidden_states
        except Exception as e:
            logger.error(f"Error generating samples: {e}")
            raise


class ConcreteHMMModel(HMMModel):
    """Concrete implementation of HMMModel with default parameters."""

    def __init__(self):
        super().__init__(n_components=4, cov_type="diag", n_iter=1000, tol=1e-4, verbose=True)
