# bot/handlers/occupancy.py
"""Occupancy pattern handlers for the bot."""
import logging
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, ContextTypes, CallbackQueryHandler
from telegram.constants import ParseMode
from bot.menu import create_back_to_main_menu_keyboard

logger = logging.getLogger(__name__)

async def show_home_patterns_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /homepatterns command to show home occupancy patterns."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized homepatterns command from user {user_id}")
        return
    
    await show_home_patterns(update.message, context)

async def show_next_event_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /nextevent command to show next expected home activity change."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized nextevent command from user {user_id}")
        return
    
    await show_next_event(update.message, context)

async def show_home_patterns(message, context, is_edit=False):
    """Show home occupancy patterns."""
    # Get occupancy analyzer from context
    occupancy_analyzer = context.application.bot_data.get("occupancy_analyzer")
    if not occupancy_analyzer:
        text = "Home activity patterns are currently unavailable."
        if is_edit:
            await message.edit_text(text, reply_markup=create_back_to_main_menu_keyboard())
        else:
            await message.reply_text(text, reply_markup=create_back_to_main_menu_keyboard())
        logger.error("Occupancy analyzer not found in bot context")
        return
    
    # Get pattern summary
    summary = occupancy_analyzer.get_pattern_summary()
    
    # Format the message
    text = "*Home Activity Patterns*\n\n"
    
    # Add last update time
    if summary.get("last_update"):
        last_update = summary["last_update"]
        text += f"📅 Last updated: {last_update}\n"
    
    # Add total patterns count
    total = summary.get("total_patterns", 0)
    text += f"📊 Total learned patterns: {total}\n\n"
    
    # Add typical empty hours by day
    text += "*Typical Empty Hours by Day:*\n"
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    empty_ranges = summary.get("empty_hour_ranges", {})
    
    for day in days:
        if day in empty_ranges and empty_ranges[day]:
            ranges_text = []
            for start, end in empty_ranges[day]:
                if start == end:
                    ranges_text.append(f"{start}:00")
                else:
                    ranges_text.append(f"{start}:00-{end}:00")
            text += f"*{day}*: Usually empty {', '.join(ranges_text)}\n"
        else:
            text += f"*{day}*: No clear pattern\n"
    
    # Create inline keyboard
    keyboard = [
        [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_home_patterns")],
        [InlineKeyboardButton("⬅️ Back to Main", callback_data="back_to_main")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if is_edit:
        await message.edit_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    else:
        await message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    
    logger.info(f"Showed home patterns for user")

async def show_next_event(message, context, is_edit=False):
    """Show next expected home activity change."""
    # Get required components from context
    occupancy_analyzer = context.application.bot_data.get("occupancy_analyzer")
    data_manager = context.application.bot_data.get("data_manager")
    
    if not occupancy_analyzer or not data_manager:
        text = "Home activity prediction is currently unavailable."
        if is_edit:
            await message.edit_text(text, reply_markup=create_back_to_main_menu_keyboard())
        else:
            await message.reply_text(text, reply_markup=create_back_to_main_menu_keyboard())
        logger.error("Missing occupancy analyzer or data manager in bot context")
        return
    
    # Get current occupancy
    current_occupants = data_manager.latest_data["room"]["occupants"]
    now = datetime.now()
    
    # Format message based on current occupancy status
    text = "*Next Home Activity Event*\n\n"
    
    if current_occupants == 0:
        # House is empty - predict next return
        next_return = occupancy_analyzer.get_next_expected_return_time(now)
        empty_duration = occupancy_analyzer.get_expected_empty_duration(now)
        
        text += "🏠 *Current Status:* Empty\n\n"
        
        if next_return:
            text += f"🔮 *Next Expected Return:* {next_return.strftime('%Y-%m-%d %H:%M')}\n"
        else:
            text += "🔮 *Next Expected Return:* Uncertain\n"
        
        if empty_duration:
            hours = empty_duration.total_seconds() / 3600
            if hours < 1:
                text += f"⏱️ *Expected Empty Duration:* {int(empty_duration.total_seconds() / 60)} minutes\n"
            else:
                text += f"⏱️ *Expected Empty Duration:* {hours:.1f} hours\n"
        else:
            text += "⏱️ *Expected Empty Duration:* Uncertain\n"
    else:
        # House is occupied - predict next departure
        text += f"🏠 *Current Status:* Occupied ({current_occupants} people)\n\n"
        text += "🔮 *Next Expected Departure:* [To be implemented]\n"
        text += "⚠️ *Note:* Departure prediction is coming soon!\n"
    
    # Create inline keyboard
    keyboard = [
        [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_next_event")],
        [InlineKeyboardButton("⬅️ Back to Main", callback_data="back_to_main")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if is_edit:
        await message.edit_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    else:
        await message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    
    logger.info(f"Showed next event for user")

async def show_home_activity_menu(query_or_message, context, is_edit=True):
    """Show home activity submenu."""
    keyboard = [
        [InlineKeyboardButton("📊 Show Patterns", callback_data="show_home_patterns")],
        [InlineKeyboardButton("🔮 Next Event", callback_data="show_next_event")],
        [InlineKeyboardButton("⬅️ Back to Main", callback_data="back_to_main")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = "*Home Activity Menu*\n\nChoose an option:"
    
    if is_edit:
        await query_or_message.edit_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
    else:
        await query_or_message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)

async def handle_occupancy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle occupancy-related callback queries."""
    query = update.callback_query
    user = query.from_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    await query.answer()
    
    if not user_auth.is_trusted(user_id):
        await query.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized occupancy callback from user {user_id}")
        return
    
    if query.data == "home_activity_menu":
        await show_home_activity_menu(query.message, context, is_edit=True)
        logger.info(f"User {user_id} accessed home activity menu")
    
    elif query.data == "show_home_patterns":
        await show_home_patterns(query.message, context, is_edit=True)
        logger.info(f"User {user_id} viewed home patterns via menu")
    
    elif query.data == "show_next_event":
        await show_next_event(query.message, context, is_edit=True)
        logger.info(f"User {user_id} viewed next event via menu")
    
    elif query.data == "refresh_home_patterns":
        # Refresh home patterns
        occupancy_analyzer = context.application.bot_data.get("occupancy_analyzer")
        if not occupancy_analyzer:
            await query.edit_message_text(
                "Home activity patterns are currently unavailable.",
                reply_markup=create_back_to_main_menu_keyboard()
            )
            return
        
        # Force reload and process history
        occupancy_analyzer._load_and_process_history()
        await show_home_patterns(query.message, context, is_edit=True)
        logger.info(f"User {user_id} refreshed home patterns")
    
    elif query.data == "refresh_next_event":
        await show_next_event(query.message, context, is_edit=True)
        logger.info(f"User {user_id} refreshed next event")

def setup_occupancy_handlers(app):
    """Register occupancy handlers."""
    app.add_handler(CommandHandler("homepatterns", show_home_patterns_command))
    app.add_handler(CommandHandler("nextevent", show_next_event_command))
    app.add_handler(CallbackQueryHandler(handle_occupancy_callback, pattern='^(home_activity_menu|show_home_patterns|show_next_event|refresh_home_patterns|refresh_next_event)$'))
    logger.info("Occupancy handlers registered")