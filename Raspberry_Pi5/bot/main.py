# bot/main.py
"""Telegram bot main entry point."""
import os
import sys
import logging
import asyncio
from telegram.ext import Application, CallbackQueryHandler

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import settings
from config.settings import BOT_TOKEN, ADMIN_ID, DATA_DIR, OCCUPANCY_HISTORY_FILE

# Import handlers with proper path resolution
from bot.handlers.commands import setup_command_handlers
from bot.handlers.messages import setup_message_handlers
from bot.user_auth import UserAuth
from bot.handlers.ventilation import setup_ventilation_handlers, handle_vent_callback
from bot.handlers.sleep_patterns import setup_sleep_handlers
from predictive.occupancy_pattern_analyzer import OccupancyPatternAnalyzer

# Import telegram ping worker
from bot.services import telegram_ping_worker

# Configure logger for the bot module
logger = logging.getLogger(__name__)
bot_file_handler = logging.FileHandler("bot.log")  # Specifies the log file for bot-specific messages.
bot_file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
bot_file_handler.setFormatter(bot_file_formatter)
bot_file_handler.setLevel(logging.INFO)
logger.addHandler(bot_file_handler)

# Global flag to stop bot gracefully
_stop_bot = False

async def heartbeat_task():
    """Periodically logs a heartbeat message to indicate the bot is operational."""
    while not _stop_bot:
        logger.info("Bot is running - heartbeat")
        await asyncio.sleep(30)  # Defines the heartbeat interval in seconds.

async def async_main(pico_manager=None, controller=None, data_manager=None, sleep_analyzer=None, preference_manager=None, occupancy_analyzer=None, device_manager=None, telegram_ping_tasks_queue=None):
    """
    Initializes and runs the asynchronous core of the Telegram bot.

    This function sets up the bot application, registers handlers for various
    Telegram events, initializes necessary service components (like user authentication,
    preference management, etc.), and starts the polling loop to receive updates.
    It also manages a heartbeat task for health monitoring and can start a
    Telegram ping worker for device connectivity checks.

    Args:
        pico_manager: Optional. Instance for managing Pico W communication.
        controller: Optional. Instance for controlling ventilation systems.
        data_manager: Optional. Instance for managing sensor data.
        sleep_analyzer: Optional. Instance for analyzing sleep patterns.
        preference_manager: Optional. Instance for managing user preferences.
        occupancy_analyzer: Optional. Instance for analyzing occupancy patterns.
        device_manager: Optional. Instance for managing connected devices.
        telegram_ping_tasks_queue: Optional. Queue for Telegram ping tasks.
    """
    global _stop_bot
    
    # Initialize user authentication module using the specified data directory.
    user_auth = UserAuth(DATA_DIR)
    
    # Create the Telegram bot application instance using the bot token.
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Initialize PreferenceManager if one is not provided externally.
    if not preference_manager:
        from preferences.preference_manager import PreferenceManager
        preference_manager = PreferenceManager()
    
    # Initialize OccupancyPatternAnalyzer if one is not provided externally.
    if not occupancy_analyzer:
        occupancy_analyzer = OccupancyPatternAnalyzer(OCCUPANCY_HISTORY_FILE)

    # Add shared components to the application context for access within handlers.
    application.bot_data["user_auth"] = user_auth
    application.bot_data["pico_manager"] = pico_manager
    application.bot_data["controller"] = controller
    application.bot_data["data_manager"] = data_manager
    application.bot_data["sleep_analyzer"] = sleep_analyzer
    application.bot_data["preference_manager"] = preference_manager
    application.bot_data["occupancy_analyzer"] = occupancy_analyzer
    application.bot_data["device_manager"] = device_manager
    
    # Setup handlers
    setup_command_handlers(application)
    setup_message_handlers(application)
    setup_ventilation_handlers(application)
    setup_sleep_handlers(application)
    from bot.handlers.preferences import setup_preference_handlers
    setup_preference_handlers(application)
    from bot.handlers.occupancy import setup_occupancy_handlers
    setup_occupancy_handlers(application)
    
    # Create and start the heartbeat task to periodically log bot health.
    heartbeat = asyncio.create_task(heartbeat_task())
    logger.info("Bot heartbeat task created")
    
    # Start polling for updates from Telegram.
    logger.info("Bot starting polling...")
    try:
        # Initialize application
        await application.initialize()
        
        # Start the updater
        await application.start()
        # Start polling; drop_pending_updates=True ensures that any messages received
        # while the bot was offline are ignored, preventing processing of stale commands.
        await application.updater.start_polling(drop_pending_updates=True)
        logger.info("Bot polling started")
        
        # Start the Telegram ping worker if its task queue and device manager are provided.
        # This worker might be used for periodic checks or keep-alive pings via Telegram.
        if telegram_ping_tasks_queue is not None and device_manager is not None:
            ping_task = asyncio.create_task(telegram_ping_worker(application.bot, device_manager, telegram_ping_tasks_queue))
            logger.info("Started Telegram ping worker")
        
        # Keep the main asynchronous loop running, periodically checking the stop signal.
        # This allows other asyncio tasks (like handlers, heartbeat) to run concurrently.
        while not _stop_bot:
            await asyncio.sleep(1)  # Periodically yield control to the event loop.
            
    except Exception as e:
        logger.error(f"Error in bot main loop: {e}", exc_info=True)
    finally:
        # Perform cleanup operations when the bot stops or an unhandled error occurs.
        logger.info("Stopping bot...")
        # Ensure application is stopped and shut down gracefully.
        # These methods handle their own state checks (e.g., if already stopped).
        await application.stop() 
        await application.shutdown() 
        
        # Cancel the heartbeat task.
        # Note: The ping_task, if created, is not explicitly cancelled in this specific finally block
        # in the original design. It would typically be cancelled when the event loop itself is closed.
        if 'heartbeat' in locals() and heartbeat and not heartbeat.done():  # Check if task exists and isn't already finished.
            heartbeat.cancel()
        logger.info("Bot stopped")

def main(pico_manager=None, controller=None, data_manager=None, sleep_analyzer=None, preference_manager=None, occupancy_analyzer=None, device_manager=None, telegram_ping_tasks_queue=None):
    """
    Main entry point to start the Telegram bot.

    This function sets up a new asyncio event loop, which is crucial if the bot
    is intended to run in a separate thread or alongside other asyncio applications.
    It then executes the `async_main` coroutine to handle the bot's lifecycle,
    including initialization, running, and error handling.
    Provides graceful shutdown on KeyboardInterrupt.

    Args:
        pico_manager: Optional. Instance for managing Pico W communication.
        controller: Optional. Instance for controlling ventilation.
        data_manager: Optional. Instance for managing sensor data.
        sleep_analyzer: Optional. Instance for analyzing sleep patterns.
        preference_manager: Optional. Instance for managing user preferences.
        occupancy_analyzer: Optional. Instance for analyzing occupancy patterns.
        device_manager: Optional. Instance for managing connected devices.
        telegram_ping_tasks_queue: Optional. Queue for Telegram ping tasks.
    """
    global _stop_bot
    
    try:
        logger.info("Starting telegram bot")
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the bot
        loop.run_until_complete(async_main(pico_manager, controller, data_manager, sleep_analyzer, preference_manager, occupancy_analyzer, device_manager, telegram_ping_tasks_queue))
        
    except Exception as e:
        logger.critical(f"Error starting bot: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Bot stopped by keyboard interrupt")
    finally:
        # Signal to stop the bot
        _stop_bot = True
        
        # Clean up the event loop and its associated resources.
        if 'loop' in locals():  # Ensure 'loop' variable was assigned.
            try:
                # Cancel all running tasks in this event loop before closing it.
                # This ensures tasks like heartbeat_task and potentially ping_task are stopped.
                all_tasks_in_loop = asyncio.all_tasks(loop)
                if all_tasks_in_loop:
                    for task in all_tasks_in_loop:
                        if not task.done():  # Only cancel tasks that aren't already completed or cancelled.
                            task.cancel()
                    # Note: The original code did not explicitly await task cancellations here.
                    # For robust shutdown, one might `loop.run_until_complete(asyncio.gather(*cancelled_tasks, return_exceptions=True))`.
                
                if not loop.is_closed():
                    loop.close()
                    logger.info("Event loop closed.")
                else:
                    # This state (already closed) would be unusual if the try block ran normally.
                    logger.info("Event loop was already closed before explicit close call in finally.")

            except RuntimeError as e:  # loop.close() can raise RuntimeError if called on a running or improperly managed loop.
                logger.error(f"RuntimeError during event loop cleanup: {e}", exc_info=True)
            except Exception as e:  # Catch any other unexpected errors during cleanup.
                logger.error(f"Error during event loop cleanup: {e}", exc_info=True)
        else:
            # This path indicates 'loop' was not defined, e.g., if an error occurred before its assignment.
            logger.warning("Event loop variable 'loop' not found in main's finally block for cleanup.")

if __name__ == "__main__":
    main()