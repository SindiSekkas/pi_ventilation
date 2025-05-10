# bot/handlers/sleep_patterns.py
"""Sleep pattern handlers for the bot."""
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, ContextTypes, CallbackQueryHandler

logger = logging.getLogger(__name__)

async def sleep_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /sleep command to show sleep analysis status."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized sleep command from user {user_id}")
        return
    
    data_manager = context.application.bot_data.get("data_manager")
    controller = context.application.bot_data.get("controller")
    sleep_analyzer = context.application.bot_data.get("sleep_analyzer")
    
    if not sleep_analyzer:
        await update.message.reply_text("Sleep pattern analysis is not available.")
        return
    
    # Get sleep pattern summary
    summary = sleep_analyzer.get_sleep_pattern_summary()
    
    # Format patterns
    patterns_text = "📊 *Detected Sleep Patterns*\n"
    weekday_patterns = summary.get("weekday_patterns", {})
    if weekday_patterns:
        for day, pattern in weekday_patterns.items():
            confidence = float(pattern.get("confidence", "0"))
            confidence_emoji = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
            patterns_text += f"{confidence_emoji} *{day}*: Sleep {pattern.get('sleep')} - Wake {pattern.get('wake')} (Conf: {pattern.get('confidence')})\n"
    else:
        patterns_text += "No sleep patterns detected yet.\n"
    
    # Format night mode
    night_mode = summary.get("current_night_mode", {})
    night_mode_text = "\n🌙 *Night Mode Settings*\n"
    night_mode_text += f"Status: {'Enabled' if night_mode.get('enabled', False) else 'Disabled'}\n"
    night_mode_text += f"Hours: {night_mode.get('start', '23:00')} - {night_mode.get('end', '07:00')}\n"
    night_mode_text += f"Currently Active: {'Yes' if night_mode.get('active', False) else 'No'}\n"
    
    # Format recent adjustments
    adjustments = summary.get("recent_adjustments", [])
    adjustments_text = "\n🔄 *Recent Adjustments*\n"
    if adjustments:
        for adj in adjustments[:3]:  # Show most recent 3
            type_text = "Sleep time" if adj.get("type") == "start_time" else "Wake time"
            adjustments_text += f"{adj.get('date')}: {type_text} {adj.get('from')} → {adj.get('to')} based on detection at {adj.get('detected_time')}\n"
    else:
        adjustments_text += "No recent adjustments made.\n"
    
    # Create full message
    message = f"*Sleep Pattern Analysis*\n\n{patterns_text}\n{night_mode_text}\n{adjustments_text}"
    
    # Add buttons
    keyboard = [
        [InlineKeyboardButton("🔄 Refresh", callback_data="sleep_refresh")],
        [InlineKeyboardButton("🌙 Night Mode Settings", callback_data="night_settings")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(message, reply_markup=reply_markup, parse_mode="Markdown")
    logger.info(f"Sleep command from user {user_id}")

async def handle_sleep_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle callback queries for sleep pattern menu."""
    query = update.callback_query
    user = query.from_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    # Always answer the callback query
    await query.answer()
    
    if not user_auth.is_trusted(user_id):
        await query.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized sleep callback from user {user_id}")
        return
    
    sleep_analyzer = context.application.bot_data.get("sleep_analyzer")
    controller = context.application.bot_data.get("controller")
    
    if query.data == "sleep_refresh":
        # Get updated sleep pattern summary
        if sleep_analyzer:
            summary = sleep_analyzer.get_sleep_pattern_summary()
            
            # Format patterns
            patterns_text = "📊 *Detected Sleep Patterns*\n"
            weekday_patterns = summary.get("weekday_patterns", {})
            if weekday_patterns:
                for day, pattern in weekday_patterns.items():
                    confidence = float(pattern.get("confidence", "0"))
                    confidence_emoji = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
                    patterns_text += f"{confidence_emoji} *{day}*: Sleep {pattern.get('sleep')} - Wake {pattern.get('wake')} (Conf: {pattern.get('confidence')})\n"
            else:
                patterns_text += "No sleep patterns detected yet.\n"
            
            # Format night mode
            night_mode = summary.get("current_night_mode", {})
            night_mode_text = "\n🌙 *Night Mode Settings*\n"
            night_mode_text += f"Status: {'Enabled' if night_mode.get('enabled', False) else 'Disabled'}\n"
            night_mode_text += f"Hours: {night_mode.get('start', '23:00')} - {night_mode.get('end', '07:00')}\n"
            night_mode_text += f"Currently Active: {'Yes' if night_mode.get('active', False) else 'No'}\n"
            
            # Format recent adjustments
            adjustments = summary.get("recent_adjustments", [])
            adjustments_text = "\n🔄 *Recent Adjustments*\n"
            if adjustments:
                for adj in adjustments[:3]:  # Show most recent 3
                    type_text = "Sleep time" if adj.get("type") == "start_time" else "Wake time"
                    adjustments_text += f"{adj.get('date')}: {type_text} {adj.get('from')} → {adj.get('to')} based on detection at {adj.get('detected_time')}\n"
            else:
                adjustments_text += "No recent adjustments made.\n"
            
            # Create full message
            message = f"*Sleep Pattern Analysis*\n\n{patterns_text}\n{night_mode_text}\n{adjustments_text}"
            
            # Update message with same buttons
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data="sleep_refresh")],
                [InlineKeyboardButton("🌙 Night Mode Settings", callback_data="night_settings")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="Markdown")
        else:
            await query.edit_message_text("Sleep pattern analysis is not available.")
    
    elif query.data == "night_settings":
        # Redirect to night settings menu
        if controller and hasattr(controller, 'night_mode_enabled'):
            # Use existing night settings menu
            from bot.handlers.ventilation import show_night_settings_menu
            await show_night_settings_menu(query, controller)
        else:
            await query.edit_message_text("Night mode settings are not available.")

def setup_sleep_handlers(app):
    """Register sleep pattern handlers."""
    app.add_handler(CommandHandler("sleep", sleep_command))
    app.add_handler(CallbackQueryHandler(handle_sleep_callback, pattern='^sleep_'))
    logger.info("Sleep pattern handlers registered")