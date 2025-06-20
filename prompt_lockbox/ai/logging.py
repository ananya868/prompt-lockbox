#
# FILE: prompt_lockbox/ai/logging.py
#
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

def setup_ai_logger(project_root: Path):
    """
    Sets up a structured JSON logger for AI interactions.
    """
    logs_dir = project_root / ".plb" / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "ai_usage.log.jsonl"

    logger = logging.getLogger("promptlockbox_ai")
    if logger.hasHandlers():
        logger.handlers.clear() # Clear existing handlers to avoid duplicates in notebooks

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    
    # Custom formatter to output logs as JSON lines
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            # --- THE FIX IS HERE ---
            # We build the base record, then update it with the 'extra' keys
            # that we know we are passing.
            log_record = {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "action": record.msg,
            }
            # Add the extra data if it exists on the record
            if hasattr(record, 'model'):
                log_record['model'] = record.model
            if hasattr(record, 'input_tokens'):
                log_record['usage'] = {
                    'input': record.input_tokens,
                    'output': record.output_tokens,
                    'total': record.total_tokens
                }
            # --- END OF FIX ---
            return json.dumps(log_record)

    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger