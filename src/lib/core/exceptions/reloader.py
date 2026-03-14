from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from .loader import ExceptionLoader


def schedule_exception_reloading(config_path: str, schema_path: str | None = None):
    """
    Schedule periodic reloading of dynamic exceptions.

    Args:
        config_path (str): Path to the exceptions configuration file.
        schema_path (Optional[str]): Path to the schema file for validation.
    """

    def reload_exceptions():
        """
        Reload dynamic exceptions using the ExceptionLoader.
        """

    try:
        loader = ExceptionLoader(
            config_path=Path(config_path), schema_path=Path(schema_path) if schema_path else Path()
        )
        loader.load_exceptions()
        print("Dynamic exceptions reloaded successfully.")
    except Exception as e:
        print(f"Error reloading exceptions: {e}")

    scheduler = BackgroundScheduler()
    scheduler.add_job(reload_exceptions, "interval", minutes=10)
    scheduler.start()
    print("Exception reloading scheduler started.")
