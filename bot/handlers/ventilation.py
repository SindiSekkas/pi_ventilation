# bot/handlers/ventilation.py
"""Ventilation control handlers for the bot."""
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, ContextTypes, CallbackQueryHandler
from bot.menu import create_main_menu, create_back_to_main_menu_keyboard, get_main_menu_message

logger = logging.getLogger(__name__)

def _get_detailed_status_text(pico_manager, controller, data_manager) -> str:
    """Generate a detailed ventilation status text including sensor readings."""
    # Get ventilation status
    current_status = pico_manager.get_ventilation_status()
    current_speed = pico_manager.get_ventilation_speed()
    
    # Get controller status
    controller_status = controller.get_status() if controller else {"auto_mode": False}
    auto_mode = controller_status.get("auto_mode", False)
    last_action = controller_status.get("last_action", "None")
    
    status_text = "📋 Ventilation Status:\n"
    status_text += f"State: {'ON' if current_status else 'OFF'}\n"
    status_text += f"Speed: {current_speed}\n"
    status_text += f"Auto Mode: {'Enabled' if auto_mode else 'Disabled'}\n"
    status_text += f"Last Auto Action: {last_action}\n"
    
    # Night mode status
    if controller and hasattr(controller, 'night_mode_enabled'):
        night_mode_info = controller_status.get("night_mode", {})
        status_text += f"Night Mode: {'Enabled' if night_mode_info.get('enabled', False) else 'Disabled'}"
        if night_mode_info.get('enabled', False):
            status_text += f" ({night_mode_info.get('start_hour', 23)}:00-{night_mode_info.get('end_hour', 7)}:00)"
        if night_mode_info.get('currently_active', False):
            status_text += " | Currently Active"
        status_text += "\n"
    
    status_text += "\n"
    
    if data_manager and hasattr(data_manager, 'latest_data'):
        scd41_data = data_manager.latest_data.get("scd41", {})
        room_data = data_manager.latest_data.get("room", {})

        co2 = scd41_data.get("co2")
        temp = scd41_data.get("temperature")
        humidity = scd41_data.get("humidity")
        occupants = room_data.get("occupants", "N/A")
        
        status_text += "📊 Current Conditions:\n"
        if co2 is not None: status_text += f"🌬️ CO2: {co2} ppm\n"
        else: status_text += "🌬️ CO2: N/A\n"
        if temp is not None: status_text += f"🌡️ Temperature: {temp}°C\n"
        else: status_text += "🌡️ Temperature: N/A\n"
        if humidity is not None: status_text += f"💧 Humidity: {humidity}%\n"
        else: status_text += "💧 Humidity: N/A\n"
        status_text += f"👥 Occupants: {occupants}\n"
    else:
        status_text += "📊 Current Conditions: Data not available\n"
        
    return status_text

async def vent_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display ventilation control menu to authorized users."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized vent command from user {user_id}")
        return
    
    pico_manager = context.application.bot_data["pico_manager"]
    controller = context.application.bot_data.get("controller")
    data_manager = context.application.bot_data.get("data_manager")
    
    auto_mode = controller.get_status()["auto_mode"] if controller else False
    
    keyboard = []
    
    auto_text = "🔴 Disable Auto Mode" if auto_mode else "🟢 Enable Auto Mode"
    keyboard.append([InlineKeyboardButton(auto_text, callback_data="vent_auto_toggle")])
    
    if controller and hasattr(controller, 'night_mode_enabled'):
        night_text = "🌙 Night Mode Settings"
        keyboard.append([InlineKeyboardButton(night_text, callback_data="vent_night_settings")])
    
    if auto_mode:
        keyboard.append([InlineKeyboardButton("⚠️ Manual Control (Auto Mode Active)", callback_data="vent_auto_notice")])
    else:
        keyboard.append([
            InlineKeyboardButton("⏹️ Off", callback_data="vent_off"),
            InlineKeyboardButton("🔽 Low", callback_data="vent_low")
        ])
        keyboard.append([
            InlineKeyboardButton("▶️ Medium", callback_data="vent_medium"),
            InlineKeyboardButton("🔼 Max", callback_data="vent_max")
        ])
    
    keyboard.append([InlineKeyboardButton("📊 Check Status", callback_data="vent_status")])
    keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="back_to_main")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    detailed_status_text = _get_detailed_status_text(pico_manager, controller, data_manager)
    
    await update.message.reply_text(
        f"🌡️ Ventilation Control\n\n{detailed_status_text}",
        reply_markup=reply_markup
    )

async def vent_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current ventilation status to authorized users."""
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
    
    status_text = _get_detailed_status_text(pico_manager, controller, data_manager)
    
    await update.message.reply_text(status_text)

async def handle_vent_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process ventilation control callback queries."""
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
                # If auto mode is enabled and we're turning it off, show confirmation dialog
                if auto_mode:
                    keyboard = [
                        [
                            InlineKeyboardButton("✅ Yes", callback_data="vent_auto_off_confirm"),
                            InlineKeyboardButton("❌ No", callback_data="vent_auto_off_cancel")
                        ]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await query.edit_message_text(
                        "Are you sure you want to turn auto mode off?",
                        reply_markup=reply_markup
                    )
                    return
                else:
                    # When turning on auto mode, no confirmation needed
                    controller.set_auto_mode(True)
                    await query.edit_message_text(
                        "✅ Auto mode enabled.\n"
                        "Returning to ventilation menu..."
                    )
                    # Show menu again after a short delay
                    if context.application.job_queue:
                        context.application.job_queue.run_once(
                            lambda context: show_vent_menu(query.message, context),
                            2
                        )
                    else:
                        await show_vent_menu(query.message, context)
            else:
                await query.edit_message_text("⚠️ Auto mode control not available.")
        
        elif action == "auto_off_confirm":
            # User confirmed turning off auto mode
            if controller:
                controller.set_auto_mode(False)
                await query.edit_message_text(
                    "❌ Auto mode disabled.\n"
                    "Returning to ventilation menu..."
                )
                # Show menu again after a short delay
                if context.application.job_queue:
                    context.application.job_queue.run_once(
                        lambda context: show_vent_menu(query.message, context),
                        2
                    )
                else:
                    await show_vent_menu(query.message, context)
            else:
                await query.edit_message_text("⚠️ Auto mode control not available.")
                
        elif action == "auto_off_cancel":
            # User canceled turning off auto mode
            await query.edit_message_text(
                "✅ Auto mode remains enabled.\n"
                "Returning to ventilation menu..."
            )
            # Show menu again after a short delay
            if context.application.job_queue:
                context.application.job_queue.run_once(
                    lambda context: show_vent_menu(query.message, context),
                    2
                )
            else:
                await show_vent_menu(query.message, context)
        
        elif action == "auto_notice":
            await query.edit_message_text(
                "⚠️ Manual control is disabled while Auto Mode is active.\n"
                "Please disable Auto Mode to control ventilation manually."
            )
        
        elif action == "night_settings":
            await show_night_settings_menu(query, controller)
        
        elif action.startswith("night_"):
            await handle_night_mode_callbacks(query, controller, action[6:])
        
        elif action == "status":
            # Show detailed status using the helper function
            data_manager = context.application.bot_data.get("data_manager")
            detailed_status_text = _get_detailed_status_text(pico_manager, controller, data_manager)
            
            keyboard = [[InlineKeyboardButton("⬅️ Back to Ventilation Menu", callback_data="vent_show_menu_from_status")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(detailed_status_text, reply_markup=reply_markup)

        elif action == "show_menu_from_status":
            # This action is called from the "Back" button on the detailed status page
            await show_vent_menu(query.message, context)
        
        elif action in ["off", "low", "medium", "max"]:
            # Check if auto mode is enabled
            auto_mode = controller.get_status()["auto_mode"] if controller else False
            if auto_mode:
                await query.edit_message_text(
                    "⚠️ Cannot manually control ventilation while Auto Mode is enabled.\n"
                    "Please disable Auto Mode first."
                )
                return
            
            # Check if night mode is active
            if controller and hasattr(controller, '_is_night_mode_active') and controller._is_night_mode_active():
                await query.edit_message_text(
                    "⚠️ Night mode is currently active.\n"
                    "Ventilation cannot be turned on during night hours."
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

async def show_night_settings_menu(query, controller):
    """Display night mode configuration options."""
    if not controller or not hasattr(controller, 'night_mode_enabled'):
        await query.edit_message_text("⚠️ Night mode settings not available.")
        return
    
    night_info = controller.get_status().get("night_mode", {})
    enabled = night_info.get("enabled", False)
    start_hour = night_info.get("start_hour", 23)
    end_hour = night_info.get("end_hour", 7)
    currently_active = night_info.get("currently_active", False)
    
    keyboard = []
    
    # Enable/Disable button
    if enabled:
        keyboard.append([InlineKeyboardButton("🌙 Disable Night Mode", callback_data="vent_night_disable")])
    else:
        keyboard.append([InlineKeyboardButton("🌟 Enable Night Mode", callback_data="vent_night_enable")])
    
    # Time setting buttons
    keyboard.append([
        InlineKeyboardButton(f"Start: {start_hour}:00", callback_data="vent_night_set_start"),
        InlineKeyboardButton(f"End: {end_hour}:00", callback_data="vent_night_set_end")
    ])
    
    # Back button
    keyboard.append([InlineKeyboardButton("⬅️ Back to Ventilation", callback_data="vent_menu")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    status_text = f"🌙 Night Mode Settings\n\n"
    status_text += f"Status: {'Enabled' if enabled else 'Disabled'}\n"
    status_text += f"Time: {start_hour}:00 - {end_hour}:00\n"
    if currently_active:
        status_text += "Night mode is currently active"
    
    await query.edit_message_text(status_text, reply_markup=reply_markup)

async def handle_night_mode_callbacks(query, controller, action):
    """Process night mode setting callbacks."""
    if not controller or not hasattr(controller, 'set_night_mode'):
        await query.edit_message_text("⚠️ Night mode settings not available.")
        return
    
    # If we're returning to night settings menu, clear any pending context
    if action == "settings":
        # Clear any pending context
        from bot.handlers.messages import night_mode_context
        user_id = query.from_user.id
        if user_id in night_mode_context:
            del night_mode_context[user_id]
    
    if action == "enable":
        controller.set_night_mode(enabled=True)
        await query.edit_message_text("✅ Night mode enabled")
        # Return to night settings menu
        await show_night_settings_menu(query, controller)
    
    elif action == "disable":
        controller.set_night_mode(enabled=False)
        await query.edit_message_text("❌ Night mode disabled")
        # Return to night settings menu
        await show_night_settings_menu(query, controller)
    
    elif action == "set_start":
        # Import the night_mode_context to store the waiting state
        from bot.handlers.messages import night_mode_context
        
        # Set context that we're waiting for a start hour
        user_id = query.from_user.id
        night_mode_context[user_id] = {
            "type": "start",
            "message": query
        }
        
        await query.edit_message_text(
            "Enter start hour (0-23):\n"
            "Reply to this message with the hour number.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("Cancel", callback_data="vent_night_settings")
            ]])
        )
    
    elif action == "set_end":
        # Import the night_mode_context to store the waiting state
        from bot.handlers.messages import night_mode_context
        
        # Set context that we're waiting for an end hour
        user_id = query.from_user.id
        night_mode_context[user_id] = {
            "type": "end",
            "message": query
        }
        
        await query.edit_message_text(
            "Enter end hour (0-23):\n"
            "Reply to this message with the hour number.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("Cancel", callback_data="vent_night_settings")
            ]])
        )

async def show_vent_menu(message, context):
    """Display ventilation control menu."""
    pico_manager = context.application.bot_data["pico_manager"]
    controller = context.application.bot_data.get("controller")
    data_manager = context.application.bot_data.get("data_manager")
    
    auto_mode = controller.get_status()["auto_mode"] if controller else False
    
    keyboard = []
    
    auto_text = "🔴 Disable Auto Mode" if auto_mode else "🟢 Enable Auto Mode"
    keyboard.append([InlineKeyboardButton(auto_text, callback_data="vent_auto_toggle")])
    
    if controller and hasattr(controller, 'night_mode_enabled'):
        night_text = "🌙 Night Mode Settings"
        keyboard.append([InlineKeyboardButton(night_text, callback_data="vent_night_settings")])
    
    if auto_mode:
        keyboard.append([InlineKeyboardButton("⚠️ Manual Control (Auto Mode Active)", callback_data="vent_auto_notice")])
    else:
        keyboard.append([
            InlineKeyboardButton("⏹️ Off", callback_data="vent_off"),
            InlineKeyboardButton("🔽 Low", callback_data="vent_low")
        ])
        keyboard.append([
            InlineKeyboardButton("▶️ Medium", callback_data="vent_medium"),
            InlineKeyboardButton("🔼 Max", callback_data="vent_max")
        ])
    
    keyboard.append([InlineKeyboardButton("📊 Check Status", callback_data="vent_status")])
    keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="back_to_main")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    detailed_status_text = _get_detailed_status_text(pico_manager, controller, data_manager)
    
    await message.edit_text(
        f"🌡️ Ventilation Control\n\n{detailed_status_text}",
        reply_markup=reply_markup
    )

def setup_ventilation_handlers(app):
    """Register ventilation control handlers with the application."""
    app.add_handler(CommandHandler("vent", vent_command))
    app.add_handler(CommandHandler("ventstatus", vent_status_command))
    app.add_handler(CallbackQueryHandler(handle_vent_callback, pattern='^vent_'))
    logger.info("Ventilation handlers registered")