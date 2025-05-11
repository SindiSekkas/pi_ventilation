"""User model and authentication management for the Telegram bot."""
import os
import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class UserAuth:
    """
    Manages user authentication, storing and verifying trusted Telegram user IDs.

    This class handles loading and saving a list of trusted users from/to a JSON file.
    It also manages a temporary state for adding new users interactively.
    """
    
    def __init__(self, data_dir: str):
        """
        Initializes the UserAuth manager.

        Sets up the path for storing trusted user data, creates the necessary
        directory if it doesn't exist, and loads any existing trusted users.

        Args:
            data_dir (str): The base directory where user data (specifically
                            the 'bot/trusted_users.json' file) will be stored.
        """
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, "bot", "trusted_users.json")
        self.trusted_users: List[int] = [] # List of trusted Telegram user IDs.
        self.adding_user_mode: bool = False # Flag indicating if the bot is in a state to add a new user.
        self.adding_user_initiator: Optional[int] = None # User ID of the admin who initiated the add user process.
        
        # Ensure the directory for storing the trusted users file exists.
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        
        # Load trusted users from the persistent storage.
        self.load_users()
    
    def load_users(self) -> None:
        """
        Loads the list of trusted user IDs from the JSON file.

        If the file doesn't exist, it starts with an empty list.
        Logs errors if loading fails.
        """
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r', encoding='utf-8') as f: # Added encoding for robustness
                    data = json.load(f)
                    self.trusted_users = data.get("trusted_users", [])
                    logger.info(f"Loaded {len(self.trusted_users)} trusted users from {self.users_file}")
            else:
                logger.info(f"Trusted users file not found at {self.users_file}. Starting with an empty list.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.users_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading trusted users from {self.users_file}: {e}", exc_info=True)
    
    def save_users(self) -> None:
        """
        Saves the current list of trusted user IDs to the JSON file.

        Logs errors if saving fails.
        """
        try:
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump({"trusted_users": self.trusted_users}, f, indent=2)
            logger.info(f"Saved {len(self.trusted_users)} trusted users to {self.users_file}")
        except Exception as e:
            logger.error(f"Error saving trusted users to {self.users_file}: {e}", exc_info=True)
    
    def is_trusted(self, user_id: int) -> bool:
        """
        Checks if a given user ID is in the list of trusted users.

        Args:
            user_id (int): The Telegram user ID to check.

        Returns:
            bool: True if the user ID is trusted, False otherwise.
        """
        return user_id in self.trusted_users
    
    def add_trusted_user(self, user_id: int) -> bool:
        """
        Adds a new user ID to the list of trusted users and saves the list.

        Args:
            user_id (int): The Telegram user ID to add.

        Returns:
            bool: True if the user was successfully added, False if the user
                  was already in the trusted list.
        """
        if user_id in self.trusted_users:
            logger.warning(f"Attempted to add user {user_id} who is already trusted.")
            return False
            
        self.trusted_users.append(user_id)
        self.save_users()
        logger.info(f"User {user_id} added to trusted users.")
        return True
    
    def start_adding_user(self, initiator_id: int) -> None:
        """
        Activates the 'adding user' mode.

        This mode is typically initiated by an existing trusted user (admin)
        to authorize a new user by having them interact with the bot.

        Args:
            initiator_id (int): The Telegram user ID of the user (admin)
                                who initiated this mode.
        """
        self.adding_user_mode = True
        self.adding_user_initiator = initiator_id
        logger.info(f"User {initiator_id} activated 'adding new user' mode.")
    
    def stop_adding_user(self) -> None:
        """Deactivates the 'adding user' mode and resets the initiator."""
        self.adding_user_mode = False
        self.adding_user_initiator = None
        logger.info("'Adding new user' mode deactivated.")
    
    def is_adding_user_mode(self) -> bool:
        """
        Checks if the bot is currently in 'adding user' mode.

        Returns:
            bool: True if in 'adding user' mode, False otherwise.
        """
        return self.adding_user_mode
    
    def process_first_user_if_needed(self, user_id: int) -> bool:
        """
        Adds the first interacting user as trusted if no trusted users exist.

        This is useful for initial bot setup, automatically trusting the first
        person who interacts with it.

        Args:
            user_id (int): The Telegram user ID of the interacting user.

        Returns:
            bool: True if this was the first user and they were added, False otherwise.
        """
        if not self.trusted_users:
            logger.info(f"No trusted users found. Adding user {user_id} as the first trusted user.")
            self.add_trusted_user(user_id) # add_trusted_user already logs and saves
            return True
        return False
