"""Telegram bot main entry point."""
import os
import sys
import logging
from telegram.ext import Application

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import settings
from config.settings import BOT_TOKEN, ADMIN_ID, DATA_DIR

# Import handlers with proper path resolution
from bot.handlers.commands import setup_command_handlers
from bot.handlers.messages import setup_message_handlers
from bot.user_auth import UserAuth

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Start the bot."""
    try:
        logger.info("Starting telegram bot")
        
        # Initialize user authentication
        user_auth = UserAuth(DATA_DIR)
        
        # Create application
        app = Application.builder().token(BOT_TOKEN).build()
        
        # Add user_auth to application context
        app.bot_data["user_auth"] = user_auth
        
        # Setup handlers
        setup_command_handlers(app)
        setup_message_handlers(app)
        
        # Start polling
        logger.info("Bot started successfully")
        app.run_polling()
        
    except Exception as e:
        logger.critical(f"Error starting bot: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()