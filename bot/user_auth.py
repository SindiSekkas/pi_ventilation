"""User model and authentication management."""
import os
import json
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class UserAuth:
    """Manages user authentication for the bot."""
    
    def __init__(self, data_dir: str):
        """Initialize user authentication manager.
        
        Args:
            data_dir: Directory to store user data
        """
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, "bot", "trusted_users.json")
        self.trusted_users: List[int] = []
        self.adding_user_mode = False
        self.adding_user_initiator: Optional[int] = None
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        
        # Load existing users
        self.load_users()
    
    def load_users(self) -> None:
        """Load trusted users from file."""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    self.trusted_users = data.get("trusted_users", [])
                    logger.info(f"Loaded {len(self.trusted_users)} trusted users")
            else:
                logger.info("No trusted users file found, starting with empty list")
        except Exception as e:
            logger.error(f"Error loading trusted users: {e}")
    
    def save_users(self) -> None:
        """Save trusted users to file."""
        try:
            with open(self.users_file, 'w') as f:
                json.dump({"trusted_users": self.trusted_users}, f, indent=2)
            logger.info(f"Saved {len(self.trusted_users)} trusted users")
        except Exception as e:
            logger.error(f"Error saving trusted users: {e}")
    
    def is_trusted(self, user_id: int) -> bool:
        """Check if a user is trusted.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            True if user is trusted, False otherwise
        """
        return user_id in self.trusted_users
    
    def add_trusted_user(self, user_id: int) -> bool:
        """Add a user to trusted users.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            True if user was added, False if already trusted
        """
        if user_id in self.trusted_users:
            return False
            
        self.trusted_users.append(user_id)
        self.save_users()
        return True
    
    def start_adding_user(self, initiator_id: int) -> None:
        """Start mode for adding a new user.
        
        Args:
            initiator_id: ID of user who initiated adding
        """
        self.adding_user_mode = True
        self.adding_user_initiator = initiator_id
        logger.info(f"User {initiator_id} started adding new user mode")
    
    def stop_adding_user(self) -> None:
        """Stop mode for adding a new user."""
        self.adding_user_mode = False
        self.adding_user_initiator = None
        logger.info("Adding new user mode stopped")
    
    def is_adding_user_mode(self) -> bool:
        """Check if bot is in adding user mode.
        
        Returns:
            True if in adding user mode, False otherwise
        """
        return self.adding_user_mode
    
    def process_first_user_if_needed(self, user_id: int) -> bool:
        """Process first user logic - if no trusted users, add first user.
        
        Args:
            user_id: Telegram user ID of potential first user
            
        Returns:
            True if this was the first user and was added, False otherwise
        """
        if not self.trusted_users:
            self.add_trusted_user(user_id)
            logger.info(f"Added first user {user_id} as trusted")
            return True
        return False
