"""Menu creation functions for the bot."""
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def create_main_menu():
    """Create the main menu keyboard."""
    keyboard = [
        [InlineKeyboardButton("👤 Add New User", callback_data="add_user")],
        [InlineKeyboardButton("🌡️ Ventilation Control", callback_data="vent_menu")],
        [InlineKeyboardButton("🌙 Sleep Analysis", callback_data="sleep_refresh")],
        [InlineKeyboardButton("⚙️ My Preferences", callback_data="my_preferences")],
        [InlineKeyboardButton("🏠 Home Activity", callback_data="home_activity_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_back_to_main_menu_keyboard():
    """Create a keyboard with just a back to main menu button."""
    keyboard = [[InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]]
    return InlineKeyboardMarkup(keyboard)

def create_back_button(callback_data):
    """Create a keyboard with just a back button."""
    keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data=callback_data)]]
    return InlineKeyboardMarkup(keyboard)

def create_yes_no_keyboard(yes_callback, no_callback, yes_text="✅ Yes", no_text="❌ No"):
    """Create a yes/no confirmation keyboard."""
    keyboard = [[
        InlineKeyboardButton(yes_text, callback_data=yes_callback),
        InlineKeyboardButton(no_text, callback_data=no_callback)
    ]]
    return InlineKeyboardMarkup(keyboard)

def get_main_menu_message(user_first_name):
    """Get the standard main menu message."""
    return f"Hi {user_first_name}! What would you like to do?"