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
    """Handle /homepatterns command."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized homepatterns command from user {user_id}")
        return
    
    await show_home_patterns(update.message, context)

async def show_next_event_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /nextevent command."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized nextevent command from user {user_id}")
        return
    
    await show_next_event(update.message, context)

async def show_home_patterns(message, context, is_edit=False):
    """Display home occupancy patterns."""
    occupancy_analyzer = context.application.bot_data.get("occupancy_analyzer")
    if not occupancy_analyzer:
        text = "Home activity patterns are currently unavailable."
        if is_edit:
            await message.edit_text(text, reply_markup=create_back_to_main_menu_keyboard())
        else:
            await message.reply_text(text, reply_markup=create_back_to_main_menu_keyboard())
        logger.error("Occupancy analyzer not found in bot context")
        return
    
    summary = occupancy_analyzer.get_pattern_summary()
    
    text = "*Home Activity Patterns*\n\n"
    
    if summary.get("last_update"):
        last_update = datetime.fromisoformat(summary["last_update"]).strftime('%Y-%m-%d %H:%M')
        text += f"📅 Last updated: {last_update}\n"
    
    total = summary.get("total_patterns", 0)
    text += f"📊 Total learned patterns: {total}\n\n"
    
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
    """Display next predicted home event."""
    occupancy_analyzer = context.application.bot_data.get("occupancy_analyzer")
    data_manager = context.application.bot_data.get("data_manager")
    
    if not occupancy_analyzer or not data_manager:
        text = "Home activity prediction is currently unavailable."
        if is_edit:
            await message.edit_text(text, reply_markup=create_back_to_main_menu_keyboard())
        else:
            await message.reply_text(text, reply_markup=create_back_to_main_menu_keyboard())
        logger.error("Missing occupancy analyzer or data manager")
        return
    
    current_occupants = data_manager.latest_data["room"]["occupants"]
    now = datetime.now()
    
    next_event = occupancy_analyzer.get_next_significant_event(now)
    current_period = occupancy_analyzer.get_predicted_current_period(now)
    
    text = "*Next Home Activity Event*\n\n"
    text += f"🏠 *Current Status:* {'Empty' if current_occupants == 0 else f'Occupied ({current_occupants} people)'}\n\n"
    
    if current_period[0] and current_period[1] and current_period[2]:
        period_start, period_end, period_status, period_confidence = current_period
        status_icon = "🏠" if period_status == "EXPECTED_OCCUPIED" else "🌙"
        text += f"{status_icon} *Current Period:* {period_status.replace('EXPECTED_', '').title()}\n"
        text += f"📅 Period: {period_start.strftime('%Y-%m-%d %H:%M')} to {period_end.strftime('%Y-%m-%d %H:%M')}\n"
        text += f"📊 Confidence: {period_confidence:.1%}\n\n"
    
    if next_event[0] and next_event[1]:
        event_time, event_type, event_confidence = next_event
        event_icon = "🏠" if event_type == "EXPECTED_ARRIVAL" else "🚪"
        text += f"{event_icon} *Next Event:* {event_type.replace('EXPECTED_', '').title()}\n"
        text += f"⏰ Time: {event_time.strftime('%Y-%m-%d %H:%M')}\n"
        text += f"📊 Confidence: {event_confidence:.1%}\n"
    else:
        text += "🔮 *Next Event:* Unable to predict with confidence\n"
    
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
    """Display home activity menu."""
    keyboard = [
        [InlineKeyboardButton("📊 Show Patterns", callback_data="show_home_patterns")],
        [InlineKeyboardButton("🔮 Next Event", callback_data="show_next_event")],
        [InlineKeyboardButton("✅ I'm Home (Correct)", callback_data="occupancy_feedback_im_home"),
         InlineKeyboardButton("❌ I'm Away (Correct)", callback_data="occupancy_feedback_im_away")],
        [InlineKeyboardButton("⬅️ Back to Main", callback_data="back_to_main")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = "*Home Activity Menu*\n\nChoose an option:"
    
    if is_edit:
        await query_or_message.edit_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
    else:
        await query_or_message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)

async def handle_occupancy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process occupancy callback queries."""
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
        occupancy_analyzer = context.application.bot_data.get("occupancy_analyzer")
        if not occupancy_analyzer:
            await query.edit_message_text(
                "Home activity patterns are currently unavailable.",
                reply_markup=create_back_to_main_menu_keyboard()
            )
            return
        
        occupancy_analyzer._load_and_process_history()
        await show_home_patterns(query.message, context, is_edit=True)
        logger.info(f"User {user_id} refreshed home patterns")
    
    elif query.data == "refresh_next_event":
        await show_next_event(query.message, context, is_edit=True)
        logger.info(f"User {user_id} refreshed next event")
    
    elif query.data == "occupancy_feedback_im_home":
        await handle_occupancy_feedback(update, context, "USER_CONFIRMED_HOME")
    
    elif query.data == "occupancy_feedback_im_away":
        await handle_occupancy_feedback(update, context, "USER_CONFIRMED_AWAY")

async def handle_occupancy_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE, feedback_status: str):
    """Process user feedback on occupancy status."""
    query = update.callback_query
    user_id = query.from_user.id
    occupancy_analyzer = context.application.bot_data.get("occupancy_analyzer")
    
    if not occupancy_analyzer:
        await query.edit_message_text(
            "⚠️ Unable to process feedback at this time.",
            reply_markup=create_back_to_main_menu_keyboard()
        )
        return
    
    now = datetime.now()
    occupancy_analyzer.record_user_feedback(now, feedback_status)
    
    feedback_text = "Thanks for the feedback! I'll update my predictions.\n\n"
    
    if feedback_status == "USER_CONFIRMED_HOME":
        feedback_text += "✅ I've noted that you're currently home."
    else:
        feedback_text += "❌ I've noted that you're currently away."
    
    feedback_text += "\n\nThis helps improve future occupancy predictions."
    
    keyboard = [
        [InlineKeyboardButton("🏠 Back to Home Activity", callback_data="home_activity_menu")],
        [InlineKeyboardButton("⬅️ Back to Main", callback_data="back_to_main")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        feedback_text,
        reply_markup=reply_markup
    )
    
    logger.info(f"User {user_id} provided occupancy feedback: {feedback_status}")

def setup_occupancy_handlers(app):
    """Register handlers."""
    app.add_handler(CommandHandler("homepatterns", show_home_patterns_command))
    app.add_handler(CommandHandler("nextevent", show_next_event_command))
    app.add_handler(CallbackQueryHandler(handle_occupancy_callback, pattern='^(home_activity_menu|show_home_patterns|show_next_event|refresh_home_patterns|refresh_next_event|occupancy_feedback_im_home|occupancy_feedback_im_away)$'))
    logger.info("Occupancy handlers registered")