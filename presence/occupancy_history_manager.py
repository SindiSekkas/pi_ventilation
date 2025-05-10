# presence/occupancy_history_manager.py
"""Manager for logging occupancy history changes."""
import os
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger(__name__)

class OccupancyHistoryManager:
    """Manages logging of occupancy status changes to CSV file."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the occupancy history manager.
        
        Args:
            data_dir: Directory to store occupancy history data
        """
        self.data_dir = data_dir
        self.csv_dir = os.path.join(data_dir, "occupancy_history")
        self.csv_file = os.path.join(self.csv_dir, "occupancy_history.csv")
        
        # Create directories if needed
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
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
        Record an occupancy status change.
        
        Args:
            status: Status ('EMPTY' or 'OCCUPIED')
            people_count: Number of people currently present
            timestamp: Timestamp of the change (default: now)
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
        Get occupancy history for the specified number of days.
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            list: List of occupancy records
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