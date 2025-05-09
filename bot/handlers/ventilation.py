"""Ventilation control handlers for the bot."""
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, ContextTypes, CallbackQueryHandler

logger = logging.getLogger(__name__)

async def vent_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /vent command to show ventilation control menu."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized vent command from user {user_id}")
        return
    
    pico_manager = context.application.bot_data["pico_manager"]
    controller = context.application.bot_data.get("controller")
    
    # Get current ventilation status
    current_status = pico_manager.get_ventilation_status()
    current_speed = pico_manager.get_ventilation_speed()
    auto_mode = controller.get_status()["auto_mode"] if controller else False
    
    # Create ventilation control menu
    keyboard = []
    
    # Add auto mode toggle
    auto_text = "Disable Auto Mode" if auto_mode else "Enable Auto Mode"
    keyboard.append([InlineKeyboardButton(auto_text, callback_data="vent_auto_toggle")])
    
    # Manual control buttons (disabled if auto mode is on)
    if auto_mode:
        keyboard.append([InlineKeyboardButton("Manual Control (Auto Mode Active)", callback_data="vent_auto_notice")])
    else:
        keyboard.append([
            InlineKeyboardButton("Off", callback_data="vent_off"),
            InlineKeyboardButton("Low", callback_data="vent_low")
        ])
        keyboard.append([
            InlineKeyboardButton("Medium", callback_data="vent_medium"),
            InlineKeyboardButton("Max", callback_data="vent_max")
        ])
    
    # Status button
    keyboard.append([InlineKeyboardButton("Check Status", callback_data="vent_status")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    status_text = f"Current: {'ON' if current_status else 'OFF'} ({current_speed})\n"
    status_text += f"Auto Mode: {'Enabled' if auto_mode else 'Disabled'}"
    
    await update.message.reply_text(
        f"Ventilation Control\n\n{status_text}",
        reply_markup=reply_markup
    )

async def vent_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ventstatus command to show current ventilation status."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized ventstatus command from user {user_id}")
        return
    
    pico_manager = context.application.bot_data["pico_manager"]
    controller = context.application.bot_data.get("controller")
    data_manager = context.application.bot_data.get("data_manager")
    
    # Get ventilation status
    current_status = pico_manager.get_ventilation_status()
    current_speed = pico_manager.get_ventilation_speed()
    
    # Get controller status
    controller_status = controller.get_status() if controller else {"auto_mode": False}
    auto_mode = controller_status["auto_mode"]
    last_action = controller_status.get("last_action", "None")
    
    # Get sensor data
    status_text = "Ventilation Status:\n"
    status_text += f"State: {'ON' if current_status else 'OFF'}\n"
    status_text += f"Speed: {current_speed}\n"
    status_text += f"Auto Mode: {'Enabled' if auto_mode else 'Disabled'}\n"
    status_text += f"Last Auto Action: {last_action}\n\n"
    
    if data_manager:
        co2 = data_manager.latest_data["scd41"]["co2"]
        temp = data_manager.latest_data["scd41"]["temperature"]
        humidity = data_manager.latest_data["scd41"]["humidity"]
        occupants = data_manager.latest_data["room"]["occupants"]
        
        status_text += "Current Conditions:\n"
        if co2: status_text += f"CO2: {co2} ppm\n"
        if temp: status_text += f"Temperature: {temp}°C\n"
        if humidity: status_text += f"Humidity: {humidity}%\n"
        status_text += f"Occupants: {occupants}\n"
    
    await update.message.reply_text(status_text)

async def handle_vent_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle ventilation control callback queries."""
    query = update.callback_query
    user = query.from_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    # Always answer the callback query
    await query.answer()
    
    if not user_auth.is_trusted(user_id):
        await query.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized vent callback from user {user_id}")
        return
    
    pico_manager = context.application.bot_data["pico_manager"]
    controller = context.application.bot_data.get("controller")
    
    if query.data.startswith("vent_"):
        action = query.data[5:]  # Remove "vent_" prefix
        
        if action == "auto_toggle":
            # Toggle auto mode
            auto_mode = controller.get_status()["auto_mode"] if controller else False
            if controller:
                controller.set_auto_mode(not auto_mode)
                new_mode = not auto_mode
                await query.edit_message_text(
                    f"Auto mode {'enabled' if new_mode else 'disabled'}.\n"
                    "Returning to ventilation menu..."
                )
                # Show menu again after a short delay
                # Check if job_queue is available, if not show the menu immediately
                if context.application.job_queue:
                    # Don't await run_once as it returns a Job object, not a coroutine
                    context.application.job_queue.run_once(
                        lambda context: show_vent_menu(query.message, context),
                        2
                    )
                else:
                    # Job queue not available, show menu directly
                    await show_vent_menu(query.message, context)
            else:
                await query.edit_message_text("Auto mode control not available.")
        
        elif action == "auto_notice":
            await query.edit_message_text(
                "Manual control is disabled while Auto Mode is active.\n"
                "Please disable Auto Mode to control ventilation manually."
            )
        
        elif action == "status":
            # Show status
            current_status = pico_manager.get_ventilation_status()
            current_speed = pico_manager.get_ventilation_speed()
            auto_mode = controller.get_status()["auto_mode"] if controller else False
            
            status_text = f"Current: {'ON' if current_status else 'OFF'} ({current_speed})\n"
            status_text += f"Auto Mode: {'Enabled' if auto_mode else 'Disabled'}"
            
            await query.edit_message_text(status_text)
        
        elif action in ["off", "low", "medium", "max"]:
            # Check if auto mode is enabled
            auto_mode = controller.get_status()["auto_mode"] if controller else False
            if auto_mode:
                await query.edit_message_text(
                    "Cannot manually control ventilation while Auto Mode is enabled.\n"
                    "Please disable Auto Mode first."
                )
                return
            
            # Perform ventilation control
            if action == "off":
                success = pico_manager.control_ventilation("off")
                message = "Ventilation turned OFF"
            else:
                success = pico_manager.control_ventilation("on", action)
                message = f"Ventilation set to {action.upper()}"
            
            if success:
                await query.edit_message_text(f"✅ {message}")
                logger.info(f"User {user_id} manually set ventilation to {action}")
            else:
                await query.edit_message_text(f"❌ Failed to {message.lower()}")
                logger.error(f"Failed to set ventilation to {action} for user {user_id}")

async def show_vent_menu(message, context):
    """Show ventilation menu after a delay."""
    pico_manager = context.application.bot_data["pico_manager"]
    controller = context.application.bot_data.get("controller")
    
    # Get current ventilation status
    current_status = pico_manager.get_ventilation_status()
    current_speed = pico_manager.get_ventilation_speed()
    auto_mode = controller.get_status()["auto_mode"] if controller else False
    
    # Create ventilation control menu
    keyboard = []
    
    # Add auto mode toggle
    auto_text = "Disable Auto Mode" if auto_mode else "Enable Auto Mode"
    keyboard.append([InlineKeyboardButton(auto_text, callback_data="vent_auto_toggle")])
    
    # Manual control buttons (disabled if auto mode is on)
    if auto_mode:
        keyboard.append([InlineKeyboardButton("Manual Control (Auto Mode Active)", callback_data="vent_auto_notice")])
    else:
        keyboard.append([
            InlineKeyboardButton("Off", callback_data="vent_off"),
            InlineKeyboardButton("Low", callback_data="vent_low")
        ])
        keyboard.append([
            InlineKeyboardButton("Medium", callback_data="vent_medium"),
            InlineKeyboardButton("Max", callback_data="vent_max")
        ])
    
    # Status button
    keyboard.append([InlineKeyboardButton("Check Status", callback_data="vent_status")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    status_text = f"Current: {'ON' if current_status else 'OFF'} ({current_speed})\n"
    status_text += f"Auto Mode: {'Enabled' if auto_mode else 'Disabled'}"
    
    await message.edit_text(
        f"Ventilation Control\n\n{status_text}",
        reply_markup=reply_markup
    )

def setup_ventilation_handlers(app):
    """Register ventilation control handlers."""
    app.add_handler(CommandHandler("vent", vent_command))
    app.add_handler(CommandHandler("ventstatus", vent_status_command))
    app.add_handler(CallbackQueryHandler(handle_vent_callback, pattern='^vent_'))
    logger.info("Ventilation handlers registered")