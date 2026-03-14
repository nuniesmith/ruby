try:
    import joblib

    HAS_JOBLIB = True
except ImportError:
    joblib = None
    HAS_JOBLIB = False

import pandas as pd

from lib.model._shims import ModelEvaluator, log_execution, logger

try:
    from sklearn.linear_model import LinearRegression, LogisticRegression

    HAS_SKLEARN = True
except ImportError:
    LogisticRegression = None
    LinearRegression = None
    HAS_SKLEARN = False

from lib.model.base.model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, max_iter=1000, task_type="binary"):
        """
        Initialize the Logistic Regression Model.

        Args:
            max_iter (int): Maximum number of iterations.
            task_type (str): Task type ('binary', 'multi-class', or 'regression').
        """
        self.task_type = task_type.lower()
        self.max_iter = max_iter

        if self.task_type in ["binary", "multi-class"]:
            self.model = LogisticRegression(max_iter=self.max_iter) if LogisticRegression is not None else None
        elif self.task_type == "regression":
            self.model = LinearRegression() if LinearRegression is not None else None
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

    @log_execution
    def build_model(self, **kwargs):
        """
        Build the Logistic Regression model.

        Args:
            **kwargs: Additional parameters for model building.

        Returns:
            self: For method chaining.
        """
        return self

    @log_execution
    def fit(self, train_data: pd.DataFrame, target_column: str, **kwargs) -> "BaseModel":
        """
        Fit the logistic regression model to the provided dataset.

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
    def train(self, X_train, y_train, **kwargs):
        """
        Train the model. For classification tasks, a Logistic Regression model
        is trained. For regression, a Linear Regression model is used.
        """
        try:
            if self.task_type in ["binary", "multi-class"]:
                logger.info("Training Logistic Regression model for classification...")
                if LogisticRegression is None:
                    raise ImportError("scikit-learn is required for LogisticRegression")
                self.model = LogisticRegression(solver="liblinear", random_state=42, max_iter=self.max_iter)
                self.model.fit(X_train, y_train)
                logger.info("Logistic Regression training completed.")
            elif self.task_type == "regression":
                logger.info("Training Linear Regression model for regression...")
                if LinearRegression is None:
                    raise ImportError("scikit-learn is required for LinearRegression")
                self.model = LinearRegression()
                self.model.fit(X_train, y_train)
                logger.info("Linear Regression training completed.")
            else:
                raise ValueError(f"Unsupported task_type: {self.task_type}")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    @log_execution
    def predict(self, X):
        """
        Generate predictions using the trained model.

        Args:
            X (array-like): Input data.

        Returns:
            array-like: Predicted outputs.
        """
        try:
            logger.info("Predicting with the model...")
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    @log_execution
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using task-specific metrics.

        Returns:
            dict: Evaluation metrics.
        """
        logger.info("Evaluating model...")
        if self.task_type in ["binary", "multi-class"]:
            return self._evaluate_classification(X_test, y_test)
        elif self.task_type == "regression":
            return self._evaluate_regression(X_test, y_test)
        else:
            raise ValueError(f"Unsupported task_type for evaluation: {self.task_type}")

    @log_execution
    def _evaluate_classification(self, X_test, y_test):
        try:
            logger.info("Evaluating classification metrics...")
            y_pred = self.model.predict(X_test)
            if ModelEvaluator is not None:
                evaluator = ModelEvaluator(y_true=y_test, y_pred=y_pred)
                metrics = evaluator.calculate_metrics()
            else:
                metrics = {"note": "ModelEvaluator not available"}
            logger.info("Classification Metrics:")
            for metric, value in metrics.items():
                if metric != "confusion_matrix":
                    if isinstance(value, (int, float)):
                        logger.info(f"{metric.capitalize()}: {value:.4f}")
                    else:
                        logger.info(f"{metric.capitalize()}: {value}")
                else:
                    logger.info(f"Confusion Matrix:\n{value}")
            return metrics
        except Exception as e:
            logger.error(f"Error during classification evaluation: {e}")
            raise

    @log_execution
    def _evaluate_regression(self, X_test, y_test):
        try:
            logger.info("Evaluating regression metrics...")
            y_pred = self.model.predict(X_test)
            if ModelEvaluator is not None:
                evaluator = ModelEvaluator(y_true=y_test, y_pred=y_pred)
                metrics = evaluator.calculate_regression_metrics()
            else:
                metrics = {"note": "ModelEvaluator not available"}
            logger.info("Regression Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric.capitalize()}: {value:.4f}")
                else:
                    logger.info(f"{metric.capitalize()}: {value}")
            return metrics
        except Exception as e:
            logger.error(f"Error during regression evaluation: {e}")
            raise

    @log_execution
    def save_model(self, file_path):
        """
        Save the trained model to a file using joblib.
        """
        try:
            if joblib is None:
                raise ImportError("joblib is required to save models")
            joblib.dump(self.model, file_path)
            logger.info(f"Model saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @log_execution
    def load_model(self, file_path):
        """
        Load a model from a file using joblib.
        """
        try:
            if joblib is None:
                raise ImportError("joblib is required to load models")
            self.model = joblib.load(file_path)
            logger.info(f"Model loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    @log_execution
    def generate_samples(self, n_samples=10):
        """
        Generate samples is not implemented for LogisticRegressionModel.
        """
        logger.info("generate_samples() not implemented for LogisticRegressionModel.")
        raise NotImplementedError("Sample generation not supported for LogisticRegressionModel.")

    def __str__(self):
        return f"{self.__class__.__name__}(task_type={self.task_type}, max_iter={self.max_iter})"


class ConcreteLogisticRegressionModel(LogisticRegressionModel):
    def __init__(self, max_iter=1000, task_type="binary"):
        super().__init__(max_iter=max_iter, task_type=task_type)
        if LogisticRegression is not None:
            self.model = LogisticRegression(max_iter=self.max_iter)

    def train(self, X_train, y_train, **kwargs):
        """
        Train the model. For classification tasks, a Logistic Regression model
        is trained. For regression, a Linear Regression model is used.
        """
        try:
            if self.task_type in ["binary", "multi-class"]:
                logger.info("Training Logistic Regression model for classification...")
                if LogisticRegression is None:
                    raise ImportError("scikit-learn is required for LogisticRegression")
                self.model = LogisticRegression(solver="liblinear", random_state=42, max_iter=self.max_iter)
                self.model.fit(X_train, y_train)
                logger.info("Logistic Regression training completed.")
            elif self.task_type == "regression":
                logger.info("Training Linear Regression model for regression...")
                if LinearRegression is None:
                    raise ImportError("scikit-learn is required for LinearRegression")
                self.model = LinearRegression()
                self.model.fit(X_train, y_train)
                logger.info("Linear Regression training completed.")
            else:
                raise ValueError(f"Unsupported task_type: {self.task_type}")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def predict(self, X):
        """
        Generate predictions using the trained model.

        Args:
            X (array-like): Input data.

        Returns:
            array-like: Predicted outputs.
        """
        try:
            logger.info("Predicting with the model...")
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using task-specific metrics.

        Returns:
            dict: Evaluation metrics.
        """
        logger.info("Evaluating model...")
        if self.task_type in ["binary", "multi-class"]:
            return self._evaluate_classification(X_test, y_test)
        elif self.task_type == "regression":
            return self._evaluate_regression(X_test, y_test)
        else:
            raise ValueError(f"Unsupported task_type for evaluation: {self.task_type}")

    def _evaluate_classification(self, X_test, y_test):
        try:
            logger.info("Evaluating classification metrics...")
            y_pred = self.model.predict(X_test)
            if ModelEvaluator is not None:
                evaluator = ModelEvaluator(y_true=y_test, y_pred=y_pred)
                metrics = evaluator.calculate_metrics()
            else:
                metrics = {"note": "ModelEvaluator not available"}
            logger.info("Classification Metrics:")
            for metric, value in metrics.items():
                if metric != "confusion_matrix":
                    if isinstance(value, (int, float)):
                        logger.info(f"{metric.capitalize()}: {value:.4f}")
                    else:
                        logger.info(f"{metric.capitalize()}: {value}")
                else:
                    logger.info(f"Confusion Matrix:\n{value}")
            return metrics
        except Exception as e:
            logger.error(f"Error during classification evaluation: {e}")
            raise
