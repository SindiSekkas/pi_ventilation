# bot/handlers/messages.py
"""Message handlers for the bot."""
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import MessageHandler, ContextTypes, filters

logger = logging.getLogger(__name__)

# Context for night mode settings
night_mode_context = {}

async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process text messages."""
    user = update.effective_user
    user_id = user.id
    text = update.message.text
    user_auth = context.application.bot_data["user_auth"]
    
    # Check if first user
    first_user = user_auth.process_first_user_if_needed(user_id)
    
    if first_user:
        await update.message.reply_text(
            f"Hi {user.first_name}! You are registered as the first trusted user for this bot."
        )
        logger.info(f"First user {user_id} registered from message")
        return
        
    # Handle add user mode
    if user_auth.is_adding_user_mode():
        if not user_auth.is_trusted(user_id):
            user_auth.add_trusted_user(user_id)
            
            await update.message.reply_text(
                f"Hi {user.first_name}! You have been added as a trusted user."
            )
            
            user_auth.stop_adding_user()
            logger.info(f"New user {user_id} added as trusted")
            return
    
    # Ignore untrusted users
    if not user_auth.is_trusted(user_id):
        logger.warning(f"Ignored message from untrusted user {user_id}")
        return
    
    # Handle night mode settings
    if user_id in night_mode_context:
        context_data = night_mode_context[user_id]
        controller = context.application.bot_data.get("controller")
        
        if not controller or not hasattr(controller, 'set_night_mode'):
            await update.message.reply_text("⚠️ Night mode settings not available.")
            del night_mode_context[user_id]
            return
        
        try:
            hour = int(text)
            if hour < 0 or hour > 23:
                await update.message.reply_text("⚠️ Please enter a valid hour (0-23).")
                return
                
            setting_type = context_data.get("type")
            
            if setting_type == "start":
                controller.set_night_mode(
                    enabled=controller.night_mode_enabled,
                    start_hour=hour,
                    end_hour=None
                )
                await update.message.reply_text(f"✅ Night mode start hour set to {hour}:00")
            elif setting_type == "end":
                controller.set_night_mode(
                    enabled=controller.night_mode_enabled,
                    start_hour=None,
                    end_hour=hour
                )
                await update.message.reply_text(f"✅ Night mode end hour set to {hour}:00")
            
            del night_mode_context[user_id]
            
            if context_data.get("message"):
                from bot.handlers.ventilation import show_night_settings_menu
                await show_night_settings_menu(context_data["message"], controller)
            
            return
            
        except ValueError:
            await update.message.reply_text("⚠️ Please enter a number between 0 and 23.")
            return
    
    # Echo for trusted users
    await update.message.reply_text(f"You said: {text}")
    logger.debug(f"Echo message from trusted user {user_id}: {text}")

def setup_message_handlers(app):
    """Register message handlers."""
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_message))
    logger.info("Message handlers registered")