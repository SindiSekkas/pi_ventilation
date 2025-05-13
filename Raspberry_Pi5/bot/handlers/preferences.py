# bot/handlers/preferences.py
"""User preference handlers."""
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, ContextTypes, CallbackQueryHandler, MessageHandler, filters
from telegram.constants import ParseMode
from bot.menu import create_main_menu, create_back_button, get_main_menu_message

logger = logging.getLogger(__name__)

async def myprefs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display user preferences."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized myprefs command from user {user_id}")
        return
    
    await show_preferences_menu(user_id, update.message, context)

async def preference_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle preference menu callback."""
    query = update.callback_query
    user = query.from_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    await query.answer()
    
    if not user_auth.is_trusted(user_id):
        await query.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized preference callback from user {user_id}")
        return
    
    await show_preferences_menu(user_id, query.message, context, is_edit=True)

async def show_preferences_menu(user_id, message_obj, context, is_edit=False):
    """Display preferences menu."""
    preference_manager = context.application.bot_data.get("preference_manager")
    if not preference_manager:
        text = "Preference system is not available."
        if is_edit:
            await message_obj.edit_text(text)
        else:
            await message_obj.reply_text(text)
        return
    
    user_first_name = None
    if hasattr(message_obj, 'chat'):
        user_first_name = message_obj.chat.first_name
    
    preference = preference_manager.get_user_preference(user_id, user_first_name)
    
    text = f"*Your Comfort Preferences*\n\n"
    text += f"🌡️ **Temperature**: {preference.temp_min}°C - {preference.temp_max}°C\n"
    text += f"🌬️ **CO₂ Threshold**: {preference.co2_threshold} ppm\n"
    text += f"💧 **Humidity**: {preference.humidity_min}% - {preference.humidity_max}%\n\n"
    text += f"**Sensitivity Settings**\n"
    text += f"Temperature: {'High' if preference.sensitivity_temp >= 1.5 else 'Low' if preference.sensitivity_temp <= 0.5 else 'Normal'}\n"
    text += f"Air Quality: {'High' if preference.sensitivity_co2 >= 1.5 else 'Low' if preference.sensitivity_co2 <= 0.5 else 'Normal'}\n"
    text += f"Humidity: {'High' if preference.sensitivity_humidity >= 1.5 else 'Low' if preference.sensitivity_humidity <= 0.5 else 'Normal'}\n"
    
    keyboard = [
        [InlineKeyboardButton("🌡️ Temperature", callback_data="pref_temp"),
         InlineKeyboardButton("🌬️ CO₂", callback_data="pref_co2")],
        [InlineKeyboardButton("💧 Humidity", callback_data="pref_humidity"),
         InlineKeyboardButton("⚙️ Sensitivity", callback_data="pref_sensitivity")],
        [InlineKeyboardButton("📊 My Feedback History", callback_data="pref_history"),
         InlineKeyboardButton("✅ All Good", callback_data="feedback_comfortable")],
        [InlineKeyboardButton("❄️ Too Cold", callback_data="feedback_too_cold"),
         InlineKeyboardButton("🔥 Too Hot", callback_data="feedback_too_hot")],
        [InlineKeyboardButton("💨 Too Dry", callback_data="feedback_too_dry"),
         InlineKeyboardButton("💦 Too Humid", callback_data="feedback_too_humid")],
        [InlineKeyboardButton("😷 Stuffy", callback_data="feedback_stuffy")],
        [InlineKeyboardButton("⬅️ Back to Main", callback_data="back_to_main")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if is_edit:
        await message_obj.edit_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    else:
        await message_obj.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    
    logger.info(f"Showed preferences for user {user_id}")

async def handle_preference_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process preference callbacks."""
    query = update.callback_query
    user = query.from_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    await query.answer()
    
    if not user_auth.is_trusted(user_id):
        await query.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized preference callback from user {user_id}")
        return
    
    preference_manager = context.application.bot_data.get("preference_manager")
    if not preference_manager:
        await query.edit_message_text("Preference system is not available.")
        return
    
    if query.data == "pref_show":
        await show_preferences_menu(user_id, query.message, context, is_edit=True)
    elif query.data == "pref_temp":
        await show_temperature_settings(query, preference_manager, user_id)
    elif query.data == "pref_co2":
        await show_co2_settings(query, preference_manager, user_id)
    elif query.data == "pref_humidity":
        await show_humidity_settings(query, preference_manager, user_id)
    elif query.data == "pref_sensitivity":
        await show_sensitivity_settings(query, preference_manager, user_id)
    elif query.data == "pref_history":
        await show_feedback_history(query, preference_manager, user_id)
    elif query.data.startswith("feedback_"):
        feedback_type = query.data.replace("feedback_", "")
        await handle_feedback(query, context, feedback_type)
    elif query.data.startswith("temp_"):
        await handle_temperature_setting(query, preference_manager, user_id)
    elif query.data.startswith("co2_"):
        await handle_co2_setting(query, preference_manager, user_id)
    elif query.data.startswith("humidity_"):
        await handle_humidity_setting(query, preference_manager, user_id)
    elif query.data.startswith("sensitivity_"):
        await handle_sensitivity_setting(query, preference_manager, user_id)

async def show_temperature_settings(query, preference_manager, user_id):
    """Display temperature settings."""
    preference = preference_manager.get_user_preference(user_id)
    
    text = f"*Temperature Preferences*\n\n"
    text += f"Current range: **{preference.temp_min}°C - {preference.temp_max}°C**\n\n"
    text += "Choose what to adjust:"
    
    keyboard = [
        [InlineKeyboardButton("📉 Lower Min Temp", callback_data="temp_min_down"),
         InlineKeyboardButton("📈 Raise Min Temp", callback_data="temp_min_up")],
        [InlineKeyboardButton("📉 Lower Max Temp", callback_data="temp_max_down"),
         InlineKeyboardButton("📈 Raise Max Temp", callback_data="temp_max_up")],
        [InlineKeyboardButton("🔄 Reset to Default", callback_data="temp_reset")],
        [InlineKeyboardButton("⬅️ Back", callback_data="pref_show")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

async def show_co2_settings(query, preference_manager, user_id):
    """Display CO2 settings."""
    preference = preference_manager.get_user_preference(user_id)
    
    text = f"*CO₂ Threshold*\n\n"
    text += f"Current threshold: **{preference.co2_threshold} ppm**\n\n"
    text += "Air quality will start to feel stuffy above this level.\n"
    text += "Typical values: 400-600 (fresh), 800-1000 (acceptable), 1200+ (poor)"
    
    keyboard = [
        [InlineKeyboardButton("📉 Decrease by 50", callback_data="co2_down_50"),
         InlineKeyboardButton("📈 Increase by 50", callback_data="co2_up_50")],
        [InlineKeyboardButton("📉 Decrease by 100", callback_data="co2_down_100"),
         InlineKeyboardButton("📈 Increase by 100", callback_data="co2_up_100")],
        [InlineKeyboardButton("🔄 Reset to Default (1000)", callback_data="co2_reset")],
        [InlineKeyboardButton("⬅️ Back", callback_data="pref_show")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

async def show_humidity_settings(query, preference_manager, user_id):
    """Display humidity settings."""
    preference = preference_manager.get_user_preference(user_id)
    
    text = f"*Humidity Preferences*\n\n"
    text += f"Current range: **{preference.humidity_min}% - {preference.humidity_max}%**\n\n"
    text += "Optimal indoor humidity is typically 30-60%"
    
    keyboard = [
        [InlineKeyboardButton("📉 Lower Min", callback_data="humidity_min_down"),
         InlineKeyboardButton("📈 Raise Min", callback_data="humidity_min_up")],
        [InlineKeyboardButton("📉 Lower Max", callback_data="humidity_max_down"),
         InlineKeyboardButton("📈 Raise Max", callback_data="humidity_max_up")],
        [InlineKeyboardButton("🔄 Reset to Default", callback_data="humidity_reset")],
        [InlineKeyboardButton("⬅️ Back", callback_data="pref_show")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

async def show_sensitivity_settings(query, preference_manager, user_id):
    """Display sensitivity settings."""
    preference = preference_manager.get_user_preference(user_id)
    
    text = f"*Sensitivity Settings*\n\n"
    text += f"Temperature Sensitivity: **{'High' if preference.sensitivity_temp >= 1.5 else 'Low' if preference.sensitivity_temp <= 0.5 else 'Normal'}**\n"
    text += f"Air Quality Sensitivity: **{'High' if preference.sensitivity_co2 >= 1.5 else 'Low' if preference.sensitivity_co2 <= 0.5 else 'Normal'}**\n"
    text += f"Humidity Sensitivity: **{'High' if preference.sensitivity_humidity >= 1.5 else 'Low' if preference.sensitivity_humidity <= 0.5 else 'Normal'}**\n\n"
    text += "Higher sensitivity means the system will react more quickly to changes."
    
    keyboard = [
        [InlineKeyboardButton("🌡️ Temperature Sensitivity", callback_data="sensitivity_temp_menu")],
        [InlineKeyboardButton("🌬️ Air Quality Sensitivity", callback_data="sensitivity_co2_menu")],
        [InlineKeyboardButton("💧 Humidity Sensitivity", callback_data="sensitivity_humidity_menu")],
        [InlineKeyboardButton("⬅️ Back", callback_data="pref_show")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

async def show_feedback_history(query, preference_manager, user_id):
    """Display user feedback history."""
    summary = preference_manager.get_preference_summary(user_id)
    recent_feedback = summary["recent_feedback"]
    
    text = f"*Your Recent Feedback*\n\n"
    if recent_feedback:
        for feedback in recent_feedback:
            timestamp = feedback["timestamp"][:16]  # Just date and time
            feedback_type = feedback["feedback_type"]
            sensor_data = feedback["sensor_data"]
            
            temp = sensor_data.get("scd41", {}).get("temperature", "N/A")
            co2 = sensor_data.get("scd41", {}).get("co2", "N/A")
            humidity = sensor_data.get("scd41", {}).get("humidity", "N/A")
            
            text += f"**{timestamp}**\n"
            text += f"Feedback: {feedback_type.replace('_', ' ').title()}\n"
            text += f"Conditions: {temp}°C, {co2} ppm, {humidity}%\n\n"
    else:
        text += "No feedback recorded yet.\n"
    
    text += f"Total feedback given: {summary['feedback_count']}"
    
    keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data="pref_show")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

async def handle_feedback(query, context, feedback_type):
    """Process user feedback on comfort."""
    user_id = query.from_user.id
    preference_manager = context.application.bot_data.get("preference_manager")
    data_manager = context.application.bot_data.get("data_manager")
    
    if not preference_manager or not data_manager:
        await query.edit_message_text("System not available.")
        return
    
    current_sensor_data = data_manager.latest_data
    preference_manager.update_preference_from_feedback(user_id, feedback_type, current_sensor_data)
    
    feedback_text = "Thank you for your feedback! "
    if feedback_type == "comfortable":
        feedback_text += "I've noted that you're comfortable with current conditions."
    elif feedback_type == "too_cold":
        feedback_text += "I'll adjust your temperature preferences to be warmer."
    elif feedback_type == "too_hot":
        feedback_text += "I'll adjust your temperature preferences to be cooler."
    elif feedback_type == "stuffy":
        feedback_text += "I'll lower your CO₂ threshold for better air quality."
    elif feedback_type == "too_dry":
        feedback_text += "I'll adjust your humidity preferences to expect more moisture."
    elif feedback_type == "too_humid":
        feedback_text += "I'll adjust your humidity preferences to expect less moisture."
    
    keyboard = [[InlineKeyboardButton("⬅️ Back to Preferences", callback_data="pref_show")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(feedback_text, reply_markup=reply_markup)

async def handle_temperature_setting(query, preference_manager, user_id):
    """Apply temperature setting changes."""
    preference = preference_manager.get_user_preference(user_id)
    
    if query.data == "temp_min_down":
        preference_manager.set_user_preference(user_id, temp_min=max(preference.temp_min - 0.5, 15.0))
    elif query.data == "temp_min_up":
        preference_manager.set_user_preference(user_id, temp_min=min(preference.temp_min + 0.5, preference.temp_max - 1.0))
    elif query.data == "temp_max_down":
        preference_manager.set_user_preference(user_id, temp_max=max(preference.temp_max - 0.5, preference.temp_min + 1.0))
    elif query.data == "temp_max_up":
        preference_manager.set_user_preference(user_id, temp_max=min(preference.temp_max + 0.5, 30.0))
    elif query.data == "temp_reset":
        preference_manager.set_user_preference(user_id, temp_min=20.0, temp_max=24.0)
    
    await show_temperature_settings(query, preference_manager, user_id)

async def handle_co2_setting(query, preference_manager, user_id):
    """Apply CO2 setting changes."""
    preference = preference_manager.get_user_preference(user_id)
    
    if query.data == "co2_down_50":
        preference_manager.set_user_preference(user_id, co2_threshold=max(preference.co2_threshold - 50, 400))
    elif query.data == "co2_up_50":
        preference_manager.set_user_preference(user_id, co2_threshold=min(preference.co2_threshold + 50, 1500))
    elif query.data == "co2_down_100":
        preference_manager.set_user_preference(user_id, co2_threshold=max(preference.co2_threshold - 100, 400))
    elif query.data == "co2_up_100":
        preference_manager.set_user_preference(user_id, co2_threshold=min(preference.co2_threshold + 100, 1500))
    elif query.data == "co2_reset":
        preference_manager.set_user_preference(user_id, co2_threshold=1000)
    
    await show_co2_settings(query, preference_manager, user_id)

async def handle_humidity_setting(query, preference_manager, user_id):
    """Apply humidity setting changes."""
    preference = preference_manager.get_user_preference(user_id)
    
    if query.data == "humidity_min_down":
        preference_manager.set_user_preference(user_id, humidity_min=max(preference.humidity_min - 5.0, 10.0))
    elif query.data == "humidity_min_up":
        preference_manager.set_user_preference(user_id, humidity_min=min(preference.humidity_min + 5.0, preference.humidity_max - 5.0))
    elif query.data == "humidity_max_down":
        preference_manager.set_user_preference(user_id, humidity_max=max(preference.humidity_max - 5.0, preference.humidity_min + 5.0))
    elif query.data == "humidity_max_up":
        preference_manager.set_user_preference(user_id, humidity_max=min(preference.humidity_max + 5.0, 80.0))
    elif query.data == "humidity_reset":
        preference_manager.set_user_preference(user_id, humidity_min=30.0, humidity_max=60.0)
    
    await show_humidity_settings(query, preference_manager, user_id)

async def handle_sensitivity_setting(query, preference_manager, user_id):
    """Apply sensitivity setting changes."""
    preference = preference_manager.get_user_preference(user_id)
    
    if query.data == "sensitivity_temp_menu":
        await show_sensitivity_temp_menu(query, preference_manager, user_id)
    elif query.data == "sensitivity_co2_menu":
        await show_sensitivity_co2_menu(query, preference_manager, user_id)
    elif query.data == "sensitivity_humidity_menu":
        await show_sensitivity_humidity_menu(query, preference_manager, user_id)
    elif query.data == "sensitivity_temp_low":
        preference_manager.set_user_preference(user_id, sensitivity_temp=0.5)
        await show_sensitivity_settings(query, preference_manager, user_id)
    elif query.data == "sensitivity_temp_normal":
        preference_manager.set_user_preference(user_id, sensitivity_temp=1.0)
        await show_sensitivity_settings(query, preference_manager, user_id)
    elif query.data == "sensitivity_temp_high":
        preference_manager.set_user_preference(user_id, sensitivity_temp=1.5)
        await show_sensitivity_settings(query, preference_manager, user_id)
    elif query.data == "sensitivity_co2_low":
        preference_manager.set_user_preference(user_id, sensitivity_co2=0.5)
        await show_sensitivity_settings(query, preference_manager, user_id)
    elif query.data == "sensitivity_co2_normal":
        preference_manager.set_user_preference(user_id, sensitivity_co2=1.0)
        await show_sensitivity_settings(query, preference_manager, user_id)
    elif query.data == "sensitivity_co2_high":
        preference_manager.set_user_preference(user_id, sensitivity_co2=1.5)
        await show_sensitivity_settings(query, preference_manager, user_id)
    elif query.data == "sensitivity_humidity_low":
        preference_manager.set_user_preference(user_id, sensitivity_humidity=0.5)
        await show_sensitivity_settings(query, preference_manager, user_id)
    elif query.data == "sensitivity_humidity_normal":
        preference_manager.set_user_preference(user_id, sensitivity_humidity=1.0)
        await show_sensitivity_settings(query, preference_manager, user_id)
    elif query.data == "sensitivity_humidity_high":
        preference_manager.set_user_preference(user_id, sensitivity_humidity=1.5)
        await show_sensitivity_settings(query, preference_manager, user_id)

async def show_sensitivity_temp_menu(query, preference_manager, user_id):
    """Display temperature sensitivity options."""
    preference = preference_manager.get_user_preference(user_id)
    
    text = f"*Temperature Sensitivity*\n\n"
    text += f"Current: **{'High' if preference.sensitivity_temp >= 1.5 else 'Low' if preference.sensitivity_temp <= 0.5 else 'Normal'}**\n\n"
    text += "Choose sensitivity level:"
    
    keyboard = [
        [InlineKeyboardButton("Low (More Tolerant)", callback_data="sensitivity_temp_low")],
        [InlineKeyboardButton("Normal", callback_data="sensitivity_temp_normal")],
        [InlineKeyboardButton("High (Less Tolerant)", callback_data="sensitivity_temp_high")],
        [InlineKeyboardButton("⬅️ Back", callback_data="pref_sensitivity")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

async def show_sensitivity_co2_menu(query, preference_manager, user_id):
    """Display CO2 sensitivity options."""
    preference = preference_manager.get_user_preference(user_id)
    
    text = f"*Air Quality Sensitivity*\n\n"
    text += f"Current: **{'High' if preference.sensitivity_co2 >= 1.5 else 'Low' if preference.sensitivity_co2 <= 0.5 else 'Normal'}**\n\n"
    text += "Choose sensitivity level:"
    
    keyboard = [
        [InlineKeyboardButton("Low (More Tolerant)", callback_data="sensitivity_co2_low")],
        [InlineKeyboardButton("Normal", callback_data="sensitivity_co2_normal")],
        [InlineKeyboardButton("High (Less Tolerant)", callback_data="sensitivity_co2_high")],
        [InlineKeyboardButton("⬅️ Back", callback_data="pref_sensitivity")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

async def show_sensitivity_humidity_menu(query, preference_manager, user_id):
    """Display humidity sensitivity options."""
    preference = preference_manager.get_user_preference(user_id)
    
    text = f"*Humidity Sensitivity*\n\n"
    text += f"Current: **{'High' if preference.sensitivity_humidity >= 1.5 else 'Low' if preference.sensitivity_humidity <= 0.5 else 'Normal'}**\n\n"
    text += "Choose sensitivity level:"
    
    keyboard = [
        [InlineKeyboardButton("Low (More Tolerant)", callback_data="sensitivity_humidity_low")],
        [InlineKeyboardButton("Normal", callback_data="sensitivity_humidity_normal")],
        [InlineKeyboardButton("High (Less Tolerant)", callback_data="sensitivity_humidity_high")],
        [InlineKeyboardButton("⬅️ Back", callback_data="pref_sensitivity")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

async def settempcomfort_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set temperature range command."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized settempcomfort command from user {user_id}")
        return
    
    preference_manager = context.application.bot_data.get("preference_manager")
    if not preference_manager:
        await update.message.reply_text("Preference system is not available.")
        return
    
    if len(context.args) != 2:
        await update.message.reply_text(
            "Usage: /settempcomfort [min] [max]\n"
            "Example: /settempcomfort 21 23"
        )
        return
    
    try:
        temp_min = float(context.args[0])
        temp_max = float(context.args[1])
        
        # Validate values
        if temp_min >= temp_max:
            await update.message.reply_text("Minimum temperature must be less than maximum temperature.")
            return
        
        if temp_min < 15 or temp_max > 30:
            await update.message.reply_text("Temperature values must be between 15°C and 30°C.")
            return
        
        preference_manager.set_user_preference(user_id, temp_min=temp_min, temp_max=temp_max)
        
        await update.message.reply_text(
            f"✅ Updated your temperature comfort range to {temp_min}°C - {temp_max}°C"
        )
        logger.info(f"User {user_id} set temperature comfort range: {temp_min}-{temp_max}°C")
        
    except ValueError:
        await update.message.reply_text("Please enter valid numbers for temperature values.")

async def setco2comfort_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set CO2 threshold command."""
    user = update.effective_user
    user_id = user.id
    user_auth = context.application.bot_data["user_auth"]
    
    if not user_auth.is_trusted(user_id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        logger.warning(f"Unauthorized setco2comfort command from user {user_id}")
        return
    
    preference_manager = context.application.bot_data.get("preference_manager")
    if not preference_manager:
        await update.message.reply_text("Preference system is not available.")
        return
    
    if len(context.args) != 1:
        await update.message.reply_text(
            "Usage: /setco2comfort [threshold]\n"
            "Example: /setco2comfort 850"
        )
        return
    
    try:
        co2_threshold = int(context.args[0])
        
        if co2_threshold < 400 or co2_threshold > 1500:
            await update.message.reply_text("CO₂ threshold must be between 400 and 1500 ppm.")
            return
        
        preference_manager.set_user_preference(user_id, co2_threshold=co2_threshold)
        
        await update.message.reply_text(
            f"✅ Updated your CO₂ comfort threshold to {co2_threshold} ppm"
        )
        logger.info(f"User {user_id} set CO2 comfort threshold: {co2_threshold} ppm")
        
    except ValueError:
        await update.message.reply_text("Please enter a valid number for CO₂ threshold.")

def setup_preference_handlers(app):
    """Register handlers."""
    app.add_handler(CommandHandler("myprefs", myprefs_command))
    app.add_handler(CommandHandler("settempcomfort", settempcomfort_command))
    app.add_handler(CommandHandler("setco2comfort", setco2comfort_command))
    app.add_handler(CallbackQueryHandler(handle_preference_callback, pattern='^(pref_|feedback_|temp_|co2_|humidity_|sensitivity_)'))
    logger.info("Preference handlers registered")