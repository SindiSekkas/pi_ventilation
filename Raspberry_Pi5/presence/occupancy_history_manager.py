# presence/occupancy_history_manager.py
"""Manager for logging occupancy history changes."""
import os
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from config.settings import OCCUPANCY_HISTORY_FILE

logger = logging.getLogger(__name__)

class OccupancyHistoryManager:
    """Records and retrieves occupancy state changes for analysis and reporting."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the occupancy history manager.

        Args:
            data_dir: Legacy parameter maintained for compatibility
        """
        self.data_dir = data_dir
        self.csv_file = OCCUPANCY_HISTORY_FILE
        
        # Ensure parent directory exists
        csv_dir = os.path.dirname(self.csv_file)
        os.makedirs(csv_dir, exist_ok=True)

        # Set up storage if needed
        self._initialize_csv()

    def _initialize_csv(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'status', 'people_count'])
                logger.info(f"Created occupancy history CSV file: {self.csv_file}")
            except Exception as e:
                logger.error(f"Error creating occupancy history CSV file: {e}")

    def record_occupancy_change(self, status: str, people_count: int, timestamp: datetime = None):
        """
        Log an occupancy state transition.

        Args:
            status: Current state ('EMPTY' or 'OCCUPIED')
            people_count: Number of detected occupants
            timestamp: Event time (defaults to current time)
        """
        if status not in ['EMPTY', 'OCCUPIED']:
            logger.error(f"Invalid occupancy status: {status}")
            return

        timestamp = timestamp or datetime.now()

        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp.isoformat(),
                    status,
                    people_count
                ])

            logger.info(f"Recorded occupancy change: {status} with {people_count} people at {timestamp.isoformat()}")
        except Exception as e:
            logger.error(f"Error recording occupancy change: {e}")

    def get_history(self, days: int = 30):
        """
        Retrieve occupancy records for the specified time period.

        Args:
            days: History window in days
            
        Returns:
            list: Chronological occupancy records within the time window
        """
        try:
            records = []
            cutoff_date = datetime.now() - timedelta(days=days)

            if os.path.exists(self.csv_file):
                with open(self.csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        timestamp = datetime.fromisoformat(row['timestamp'])
                        if timestamp >= cutoff_date:
                            records.append({
                                'timestamp': timestamp,
                                'status': row['status'],
                                'people_count': int(row['people_count'])
                            })

            return records
        except Exception as e:
            logger.error(f"Error retrieving occupancy history: {e}")
            return []