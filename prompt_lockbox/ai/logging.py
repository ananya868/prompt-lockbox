#
# FILE: prompt_lockbox/ai/logging.py
#
import json
import logging
from datetime import datetime, timezone
from pathlib import Path


def setup_ai_logger(project_root: Path):
    """Sets up and configures a structured JSON logger for AI interactions.

    This function creates a dedicated logger that writes to a `.jsonl` file
    within the project's `.plb/logs` directory. Each log entry is a single
    line of JSON, making it easy to parse and analyze AI usage programmatically.
    It clears existing handlers to prevent duplicate log entries when run
    repeatedly, for instance in a Jupyter notebook.

    Args:
        project_root: The root `Path` of the PromptLockbox project.

    Returns:
        A configured `logging.Logger` instance ready for use.
    """
    # Define the path for AI usage logs within the project's hidden directory.
    logs_dir = project_root / ".plb" / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "ai_usage.log.jsonl"

    logger = logging.getLogger("promptlockbox_ai")
    # To prevent duplicate log entries in interactive environments (like notebooks),
    # clear any handlers that may have been attached in previous runs.
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)

    class JsonFormatter(logging.Formatter):
        """A custom logging formatter that outputs log records as JSON strings."""

        def format(self, record: logging.LogRecord) -> str:
            """Formats a log record into a single-line JSON string.

            This method constructs a dictionary from the log record's attributes,
            including standard fields and any custom data passed in the `extra`
            dictionary of a logging call.

            Args:
                record: The `logging.LogRecord` instance to format.

            Returns:
                A JSON-formatted string representing the log record.
            """
            # Start with the base information for every log entry.
            log_record = {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "action": record.msg,
            }
            # Safely check for and add the extra data passed to the logger call.
            # This allows for a flexible but structured logging schema.
            if hasattr(record, 'model'):
                log_record['model'] = record.model
            if hasattr(record, 'input_tokens'):
                log_record['usage'] = {
                    'input': record.input_tokens,
                    'output': record.output_tokens,
                    'total': record.total_tokens
                }

            return json.dumps(log_record)

    # Apply the custom JSON formatter to the handler and attach it to the logger.
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger