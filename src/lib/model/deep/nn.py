"""
Simple Neural Network models for time series forecasting.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np

from lib.model._shims import log_execution, logger
from lib.model.base.model import BaseModel

if TYPE_CHECKING:
    import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None


class SimpleNN(BaseModel):
    def __init__(self, input_size, hidden_size=10, output_size=1, learning_rate=0.01):
        """
        Initialize the SimpleNN.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output neurons.
            learning_rate (float): Learning rate for weight updates.
        """
        self.learning_rate = learning_rate
        # Initialize weights with small random values.
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))
        logger.info(
            f"SimpleNN initialized with input size {input_size}, hidden size {hidden_size}, output size {output_size}."
        )

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def forward(self, X):
        if not isinstance(X, np.ndarray):
            logger.error("Input to forward pass must be a numpy array.")
            raise ValueError("Input must be a numpy array.")
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_layer_input)
        logger.debug("Forward pass completed.")
        return self.output

    def backward(self, X, y, output):
        error = y - output
        output_gradient = error * self.sigmoid_derivative(output)
        hidden_error = output_gradient.dot(self.weights_hidden_output.T)
        hidden_gradient = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases.
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_gradient) * self.learning_rate
        self.bias_output += np.sum(output_gradient, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_gradient) * self.learning_rate
        self.bias_hidden += np.sum(hidden_gradient, axis=0, keepdims=True) * self.learning_rate
        logger.debug("Backward pass and weight update completed.")

    @log_execution
    def build_model(self, **kwargs):
        """
        Build the neural network model.

        Args:
            **kwargs: Additional parameters for model building.

        Returns:
            self: For method chaining.
        """
        return self

    @log_execution
    def fit(self, train_data: pd.DataFrame, target_column: str, **kwargs) -> BaseModel:
        """
        Fit the NN model to the provided dataset.

        Args:
            train_data: Data to train on
            target_column: Name of the target column
            **kwargs: Additional parameters for training

        Returns:
            self: For method chaining
        """
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]
        self.train(X_train, y_train, **kwargs)
        return self

    @log_execution
    def train(self, X, y, epochs=1000, batch_size=None, log_interval=100, **kwargs):
        """
        Train the neural network on input X and target y.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Target data.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training. If None, use full batch.
            log_interval (int): Logging frequency (in epochs).
        """
        if not (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            logger.error("Training inputs X and y must be numpy arrays.")
            raise ValueError("Training inputs X and y must be numpy arrays.")
        if len(X) != len(y):
            logger.error("X and y must have the same length.")
            raise ValueError("X and y must have the same length.")

        logger.info(f"Starting training for {epochs} epochs.")
        num_samples = len(X)
        for epoch in range(epochs):
            if batch_size:
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                for i in range(0, num_samples, batch_size):
                    batch_indices = indices[i : i + batch_size]
                    X_batch, y_batch = X[batch_indices], y[batch_indices]
                    output = self.forward(X_batch)
                    self.backward(X_batch, y_batch, output)
            else:
                output = self.forward(X)
                self.backward(X, y, output)
            if epoch % log_interval == 0:
                loss = np.mean((y - self.forward(X)) ** 2)
                logger.info(f"Epoch {epoch}: Loss: {loss:.4f}")
        logger.info("Training completed.")

    @log_execution
    def predict(self, X):
        """
        Predict using the neural network.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted labels (binary classification: 0 or 1).
        """
        if not isinstance(X, np.ndarray):
            logger.error("Input to predict must be a numpy array.")
            raise ValueError("Input must be a numpy array.")
        output = self.forward(X)
        predictions = (output > 0.5).astype(int)
        logger.info("Prediction completed.")
        return predictions

    @log_execution
    def save_model(self, file_path):
        """
        Save the model parameters to a file using pickle.
        """
        try:
            with open(file_path, "wb") as f:
                pickle.dump(
                    {
                        "weights_input_hidden": self.weights_input_hidden,
                        "bias_hidden": self.bias_hidden,
                        "weights_hidden_output": self.weights_hidden_output,
                        "bias_output": self.bias_output,
                        "learning_rate": self.learning_rate,
                    },
                    f,
                )
            logger.info(f"Model saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @log_execution
    def load_model(self, file_path):
        """
        Load model parameters from a file using pickle.
        """
        try:
            with open(file_path, "rb") as f:
                params = pickle.load(f)  # noqa: S301
            self.weights_input_hidden = params["weights_input_hidden"]
            self.bias_hidden = params["bias_hidden"]
            self.weights_hidden_output = params["weights_hidden_output"]
            self.bias_output = params["bias_output"]
            self.learning_rate = params.get("learning_rate", self.learning_rate)
            logger.info(f"Model loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    @log_execution
    def generate_samples(self, n_samples=10):
        """
        Sample generation is not supported for SimpleNN.
        """
        logger.info("generate_samples() not implemented for SimpleNN.")
        raise NotImplementedError("Sample generation is not supported for SimpleNN.")

    def __str__(self):
        return (
            f"SimpleNN(input_size={self.weights_input_hidden.shape[0]}, "
            f"hidden_size={self.weights_input_hidden.shape[1]})"
        )


class EnhancedNN(SimpleNN):
    def __init__(self, input_size, hidden_size=10, output_size=1, learning_rate=0.01):
        super().__init__(input_size, hidden_size, output_size, learning_rate)
        logger.info("EnhancedNN initialized without internal feature scaler.")

    @log_execution
    def train_with_features(self, X, y, epochs=1000, batch_size=None, log_interval=100, **kwargs):
        """
        Train the enhanced NN. Expects input X to be preprocessed.
        """
        super().train(X, y, epochs=epochs, batch_size=batch_size, log_interval=log_interval, **kwargs)

    @log_execution
    def predict_with_features(self, X):
        """
        Predict using enhanced NN. Expects input X to be preprocessed.
        """
        return super().predict(X)


class ConcreteNN(EnhancedNN):
    def __init__(self, input_size, hidden_size=10, output_size=1, learning_rate=0.01):
        super().__init__(input_size, hidden_size, output_size, learning_rate)
        if StandardScaler is not None:
            self.feature_scaler = StandardScaler()
        else:
            self.feature_scaler = None
        logger.info("ConcreteNN initialized with internal feature scaler.")

    def preprocess_features(self, X):
        """
        Preprocess input features using the internal feature scaler.

        Args:
            X (np.ndarray): Input features to preprocess.

        Returns:
            np.ndarray: Preprocessed features.
        """
        if not isinstance(X, np.ndarray):
            logger.error("Input to preprocess_features must be a numpy array.")
            raise ValueError("Input must be a numpy array.")
        if self.feature_scaler is None:
            return X
        return self.feature_scaler.transform(X)

    def train(self, X, y, epochs=1000, batch_size=None, log_interval=100, **kwargs):
        """
        Train the neural network on input X and target y.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Target data.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training. If None, use full batch.
            log_interval (int): Logging frequency (in epochs).
        """
        if not (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            logger.error("Training inputs X and y must be numpy arrays.")
            raise ValueError("Training inputs X and y must be numpy arrays.")
        if len(X) != len(y):
            logger.error("X and y must have the same length.")
            raise ValueError("X and y must have the same length.")

        logger.info(f"Starting training for {epochs} epochs.")
        X = self.preprocess_features(X)
        num_samples = len(X)
        for epoch in range(epochs):
            if batch_size:
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                for i in range(0, num_samples, batch_size):
                    batch_indices = indices[i : i + batch_size]
                    X_batch, y_batch = X[batch_indices], y[batch_indices]
                    output = self.forward(X_batch)
                    self.backward(X_batch, y_batch, output)
            else:
                output = self.forward(X)
                self.backward(X, y, output)
            if epoch % log_interval == 0:
                loss = np.mean((y - self.forward(X)) ** 2)
                logger.info(f"Epoch {epoch}: Loss: {loss:.4f}")
        logger.info("Training completed.")

    def predict(self, X):
        """
        Predict using the neural network.
        """
        X = self.preprocess_features(X)
        return super().predict(X)
