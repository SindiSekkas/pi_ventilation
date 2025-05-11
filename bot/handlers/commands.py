# bot/handlers/commands.py
"""Command handlers for the Telegram bot."""
import logging
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, ContextTypes, CallbackQueryHandler
from bot.menu import create_main_menu, create_back_to_main_menu_keyboard, get_main_menu_message

logger = logging.getLogger(__name__)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    # Check if first user
    first_user = user_auth.process_first_user_if_needed(user_id)
    
    if first_user:
        await update.message.reply_text(
            f"Hi {user.first_name}! You are registered as the first trusted user for this bot."
        )
        logger.info(f"First user {user_id} registered")
    elif user_auth.is_trusted(user_id):
        # Menu for trusted users
        reply_markup = create_main_menu()
        await update.message.reply_text(
            get_main_menu_message(user.first_name),
            reply_markup=reply_markup
        )
        logger.info(f"Start command from trusted user {user_id}")
    else:
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
        /sleep - Show sleep pattern analysis and night mode settings
        /myprefs - View and adjust your comfort preferences
        /settempcomfort [min] [max] - Set temperature comfort range
        /setco2comfort [threshold] - Set CO2 comfort threshold
        /homepatterns - Show home occupancy patterns
        /nextevent - Show next expected home activity change
        /linkphone [MAC-address] - Link your phone for presence detection
        /unlinkphone [MAC-address] - Unlink your phone from presence detection
        /pingphone [MAC-address] - Test Telegram ping for your phone

        Ventilation commands:
        /vent - Show ventilation control menu
        /ventstatus - Show current ventilation status

        Ventilation Control:
        - Turn ventilation on/off manually
        - Set ventilation speed (low/medium/max)
        - Toggle auto mode on/off
        - View current status and sensor readings

        Sleep Analysis:
        - View detected sleep patterns
        - Check night mode settings
        - See recent sleep time adjustments

        Home Activity:
        - View occupancy patterns by day and hour
        - See next expected arrival/departure
        - Get predicted empty durations

        Preferences:
        - Set your ideal temperature range
        - Configure CO2 threshold
        - Adjust humidity preferences
        - Provide comfort feedback

        Phone Linking:
        - Link your phone to get better presence detection
        - Phone will receive silent wake-up pings when offline
        - Better accuracy for occupancy tracking
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
    
    # Cancel button
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
    
    if user_auth.is_adding_user_mode():
        user_auth.stop_adding_user()
        await update.message.reply_text("Operation cancelled.")
        logger.info(f"User {user_id} cancelled add user process")
    else:
        await update.message.reply_text("No active operation to cancel.")

async def linkphone_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /linkphone command."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized linkphone attempt from user {user_id}")
        return
    
    if len(context.args) != 1:
        await update.message.reply_text(
            "Usage: /linkphone <MAC-address>\n"
            "Example: /linkphone aa:bb:cc:dd:ee:ff"
        )
        return
    
    mac_address = context.args[0].lower()
    
    # Validate MAC format
    if not re.match(r'^([0-9a-f]{2}[:-]){5}([0-9a-f]{2})$', mac_address):
        await update.message.reply_text("Please enter a valid MAC address (e.g., aa:bb:cc:dd:ee:ff)")
        return
    
    device_manager = context.application.bot_data.get("device_manager")
    if not device_manager:
        await update.message.reply_text("Device management is not available.")
        return
    
    # Find device
    found_device = None
    for mac, device in device_manager.devices.items():
        if mac.lower() == mac_address.lower():
            found_device = device
            break
    
    if not found_device:
        await update.message.reply_text(
            f"Device with MAC {mac_address} not found in the system.\n"
            "Please ensure your phone is currently connected to the network first."
        )
        return
    
    # Check device type
    if found_device.device_type != "phone":
        await update.message.reply_text(
            f"Device {mac_address} is not a phone (it's a {found_device.device_type}).\n"
            "You can only link phone devices."
        )
        return
    
    # Link device
    success = device_manager.link_device_to_telegram_user(mac_address, user_id)
    
    if success:
        await update.message.reply_text(
            f"✅ Your phone with MAC {mac_address} has been linked to your Telegram account for presence checks.\n"
            "You may receive silent notifications when your phone appears offline to help improve presence detection."
        )
        logger.info(f"User {user_id} linked phone {mac_address}")
    else:
        await update.message.reply_text("Failed to link your phone. Please try again later.")
        logger.error(f"Failed to link phone {mac_address} for user {user_id}")

async def unlinkphone_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /unlinkphone command."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized unlinkphone attempt from user {user_id}")
        return
    
    if len(context.args) != 1:
        await update.message.reply_text(
            "Usage: /unlinkphone <MAC-address>\n"
            "Example: /unlinkphone aa:bb:cc:dd:ee:ff"
        )
        return
    
    mac_address = context.args[0].lower()
    
    # Validate MAC format
    if not re.match(r'^([0-9a-f]{2}[:-]){5}([0-9a-f]{2})$', mac_address):
        await update.message.reply_text("Please enter a valid MAC address (e.g., aa:bb:cc:dd:ee:ff)")
        return
    
    device_manager = context.application.bot_data.get("device_manager")
    if not device_manager:
        await update.message.reply_text("Device management is not available.")
        return
    
    # Find device
    found_device = None
    for mac, device in device_manager.devices.items():
        if mac.lower() == mac_address.lower():
            found_device = device
            break
    
    if not found_device:
        await update.message.reply_text(f"Device with MAC {mac_address} not found in the system.")
        return
    
    if found_device.telegram_user_id != user_id:
        await update.message.reply_text(f"Device {mac_address} is not linked to your account.")
        return
    
    # Unlink device
    success = device_manager.unlink_device_from_telegram_user(mac_address)
    
    if success:
        await update.message.reply_text(
            f"✅ Your phone with MAC {mac_address} has been unlinked from your Telegram account.\n"
            "You will no longer receive notifications for this device."
        )
        logger.info(f"User {user_id} unlinked phone {mac_address}")
    else:
        await update.message.reply_text("Failed to unlink your phone. Please try again later.")
        logger.error(f"Failed to unlink phone {mac_address} for user {user_id}")

async def ping_phone_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test Telegram ping to phone."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized pingphone command from user {user_id}")
        return
    
    if len(context.args) != 1:
        await update.message.reply_text(
            "Usage: /pingphone <MAC-address>\\n"
            "Example: /pingphone aa:bb:cc:dd:ee:ff"
        )
        return
    
    mac_address = context.args[0].lower()
    
    # Validate MAC format
    if not re.match(r'^([0-9a-f]{2}[:-]){5}([0-9a-f]{2})$', mac_address):
        await update.message.reply_text("Please enter a valid MAC address (e.g., aa:bb:cc:dd:ee:ff)")
        return
    
    device_manager = context.application.bot_data.get("device_manager")
    if not device_manager:
        await update.message.reply_text("Device management is not available.")
        return
    
    # Find device
    found_device = None
    for mac, device in device_manager.devices.items():
        if mac.lower() == mac_address.lower():
            found_device = device
            break
    
    if not found_device:
        await update.message.reply_text(f"Device with MAC {mac_address} not found in the system.")
        return
    
    if found_device.telegram_user_id != user_id:
        await update.message.reply_text(f"Device {mac_address} is not linked to your account.")
        return
    
    # Queue ping task
    telegram_ping_queue = context.application.bot_data.get("telegram_ping_tasks_queue")
    if telegram_ping_queue:
        ping_task = {
            'mac': found_device.mac,
            'telegram_user_id': user_id,
            'ip_address': found_device.last_ip or None
        }
        telegram_ping_queue.put(ping_task)
        await update.message.reply_text("🔔 Telegram ping sent to your phone. Check your notifications!")
        logger.info(f"Manual Telegram ping requested for {mac_address} by user {user_id}")
    else:
        await update.message.reply_text("Telegram ping queue is not available.")

async def handle_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks."""
    query = update.callback_query
    user = query.from_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    await query.answer()
    
    if not user_auth.is_trusted(user_id):
        await query.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized button press from user {user_id}")
        return
    
    if query.data == "add_user":
        user_auth.start_adding_user(user_id)
        
        keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data="cancel_add_user")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text="The next user who sends a message will be added as a trusted user.\n"
                 "Press 'Cancel' to cancel this operation.",
            reply_markup=reply_markup
        )
        logger.info(f"User {user_id} started add user process via button")
        
    elif query.data == "cancel_add_user":
        user_auth.stop_adding_user()
        
        reply_markup = create_main_menu()
        await query.edit_message_text(
            text=f"Operation cancelled. {get_main_menu_message(user.first_name)}",
            reply_markup=reply_markup
        )
        logger.info(f"User {user_id} cancelled add user process via button")
    
    elif query.data == "back_to_main":
        reply_markup = create_main_menu()
        await query.edit_message_text(
            text=get_main_menu_message(user.first_name),
            reply_markup=reply_markup
        )
        logger.info(f"User {user_id} returned to main menu")
    
    elif query.data == "vent_menu":
        from bot.handlers.ventilation import show_vent_menu
        await show_vent_menu(query.message, context)
        logger.info(f"User {user_id} opened ventilation menu")
        
    elif query.data == "sleep_refresh":
        from bot.handlers.sleep_patterns import handle_sleep_callback
        await handle_sleep_callback(update, context)
        logger.info(f"User {user_id} refreshed sleep analysis menu")
        
    elif query.data == "night_settings":
        from bot.handlers.sleep_patterns import handle_sleep_callback
        await handle_sleep_callback(update, context)
        logger.info(f"User {user_id} accessed night settings from sleep menu")
        
    elif query.data == "my_preferences":
        from bot.handlers.preferences import preference_callback
        await preference_callback(update, context)
        logger.info(f"User {user_id} accessed preferences menu")
    
    elif query.data == "home_activity_menu":
        from bot.handlers.occupancy import show_home_activity_menu
        await show_home_activity_menu(query.message, context, is_edit=True)
        logger.info(f"User {user_id} accessed home activity menu")

def setup_command_handlers(app):
    """Register command handlers."""
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("adduser", add_user_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CommandHandler("linkphone", linkphone_command))
    app.add_handler(CommandHandler("unlinkphone", unlinkphone_command))
    app.add_handler(CommandHandler("pingphone", ping_phone_command))
    app.add_handler(CallbackQueryHandler(handle_button_callback, pattern='^(add_user|cancel_add_user|vent_menu|sleep_refresh|night_settings|my_preferences|back_to_main)$'))
    logger.info("Command handlers registered")