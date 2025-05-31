import logging
from datetime import datetime
import os

# Generate a timestamped log file name
log_filename = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the full path to the logs directory and ensure it exists
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# Full path to the log file
log_filepath = os.path.join(log_dir, log_filename)

# Configure logging
logging.basicConfig(
    filename=log_filepath,
    level=logging.INFO,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    logging.info("Logging started.")
