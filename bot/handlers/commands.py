"""Command handlers for the bot."""
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, ContextTypes, CallbackQueryHandler

logger = logging.getLogger(__name__)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    # Check if this is the first user
    first_user = user_auth.process_first_user_if_needed(user_id)
    
    if first_user:
        await update.message.reply_text(
            f"Hi {user.first_name}! You are registered as the first trusted user for this bot."
        )
        logger.info(f"First user {user_id} registered")
    elif user_auth.is_trusted(user_id):
        # Show menu for trusted users
        keyboard = [
            [InlineKeyboardButton("👤 Add New User", callback_data="add_user")],
            [InlineKeyboardButton("🌡️ Ventilation Control", callback_data="vent_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            f"Hi {user.first_name}! What would you like to do?",
            reply_markup=reply_markup
        )
        logger.info(f"Start command from trusted user {user_id}")
    else:
        # Politely reject untrusted users
        await update.message.reply_text(
            "Sorry, you are not authorized to use this bot."
        )
        logger.warning(f"Unauthorized access attempt from user {user_id}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized help access attempt from user {user_id}")
        return
    
    help_text = """
Available commands:
/start - Start bot and show main menu
/help - Show this help
/adduser - Start the process to add a new trusted user
/cancel - Cancel the current operation

Ventilation commands:
/vent - Show ventilation control menu
/ventstatus - Show current ventilation status

Ventilation Control:
- Turn ventilation on/off manually
- Set ventilation speed (low/medium/max)
- Toggle auto mode on/off
- View current status and sensor readings
    """
    await update.message.reply_text(help_text)
    logger.info(f"Help command from user {user_id}")

async def add_user_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /adduser command."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized adduser attempt from user {user_id}")
        return
    
    user_auth.start_adding_user(user_id)
    
    # Create cancel button
    keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data="cancel_add_user")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "The next user who sends a message will be added as a trusted user.\n"
        "Press 'Cancel' to cancel this operation.",
        reply_markup=reply_markup
    )
    logger.info(f"User {user_id} started add user process")

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /cancel command."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized cancel attempt from user {user_id}")
        return
    
    # Cancel add user mode if active
    if user_auth.is_adding_user_mode():
        user_auth.stop_adding_user()
        await update.message.reply_text("Operation cancelled.")
        logger.info(f"User {user_id} cancelled add user process")
    else:
        await update.message.reply_text("No active operation to cancel.")

async def handle_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callback queries."""
    query = update.callback_query
    user = query.from_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    # Always answer the callback query
    await query.answer()
    
    if not user_auth.is_trusted(user_id):
        await query.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized button press from user {user_id}")
        return
    
    if query.data == "add_user":
        # Start add user process
        user_auth.start_adding_user(user_id)
        
        # Update message with cancel button
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data="cancel_add_user")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text="The next user who sends a message will be added as a trusted user.\n"
                 "Press 'Cancel' to cancel this operation.",
            reply_markup=reply_markup
        )
        logger.info(f"User {user_id} started add user process via button")
        
    elif query.data == "cancel_add_user":
        # Cancel add user process
        user_auth.stop_adding_user()
        
        # Return to main menu
        keyboard = [
            [InlineKeyboardButton("👤 Add New User", callback_data="add_user")],
            [InlineKeyboardButton("🌡️ Ventilation Control", callback_data="vent_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text=f"Operation cancelled. What would you like to do?",
            reply_markup=reply_markup
        )
        logger.info(f"User {user_id} cancelled add user process via button")
    
    elif query.data == "vent_menu":
        from bot.handlers.ventilation import show_vent_menu
        await show_vent_menu(query.message, context)
        logger.info(f"User {user_id} opened ventilation menu")

def setup_command_handlers(app):
    """Register all command handlers."""
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("adduser", add_user_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CallbackQueryHandler(handle_button_callback, pattern='^(add_user|cancel_add_user|vent_menu)$'))
    logger.info("Command handlers registered")