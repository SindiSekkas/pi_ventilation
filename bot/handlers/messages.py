"""Message handlers for the bot."""
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import MessageHandler, ContextTypes, filters

logger = logging.getLogger(__name__)

async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Echo text messages."""
    user = update.effective_user
    user_id = user.id
    text = update.message.text
    user_auth = context.application.bot_data["user_auth"]
    
    # Check if this is the first user
    first_user = user_auth.process_first_user_if_needed(user_id)
    
    if first_user:
        await update.message.reply_text(
            f"Hi {user.first_name}! You are registered as the first trusted user for this bot."
        )
        logger.info(f"First user {user_id} registered from message")
        return
        
    # Check if we're in add user mode
    if user_auth.is_adding_user_mode():
        # Check if this is a new user (not already trusted)
        if not user_auth.is_trusted(user_id):
            # Add the new user
            user_auth.add_trusted_user(user_id)
            
            # Notify the user they've been added
            await update.message.reply_text(
                f"Hi {user.first_name}! You have been added as a trusted user."
            )
            
            # Stop add user mode
            user_auth.stop_adding_user()
            logger.info(f"New user {user_id} added as trusted")
            return
    
    # Ignore messages from untrusted users
    if not user_auth.is_trusted(user_id):
        # Silently ignore
        logger.warning(f"Ignored message from untrusted user {user_id}")
        return
    
    # Normal message handling for trusted users
    await update.message.reply_text(f"You said: {text}")
    logger.debug(f"Echo message from trusted user {user_id}: {text}")

def setup_message_handlers(app):
    """Register all message handlers."""
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_message))
    logger.info("Message handlers registered")