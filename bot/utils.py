"""Utility functions for the bot."""
from datetime import datetime

def get_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def is_admin(user_id, admin_id):
    """Check if user is admin."""
    return str(user_id) == str(admin_id)