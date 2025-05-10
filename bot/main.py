"""Telegram bot main entry point."""
# bot/main.py
import os
import sys
import logging
import asyncio
import threading
from telegram.ext import Application, CallbackQueryHandler

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import settings
from config.settings import BOT_TOKEN, ADMIN_ID, DATA_DIR

# Import handlers with proper path resolution
from bot.handlers.commands import setup_command_handlers
from bot.handlers.messages import setup_message_handlers
from bot.user_auth import UserAuth
from bot.handlers.ventilation import setup_ventilation_handlers, handle_vent_callback
from bot.handlers.sleep_patterns import setup_sleep_handlers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def async_main(pico_manager=None, controller=None, data_manager=None, sleep_analyzer=None):
    """Async main bot function."""
    # Initialize user authentication
    user_auth = UserAuth(DATA_DIR)
    
    # Create application with proper configuration
    application_kwargs = {'token': BOT_TOKEN}
    # Default timeouts
    application_kwargs['connect_timeout'] = 15.0
    application_kwargs['pool_timeout'] = 10.0
    application_kwargs['read_timeout'] = 10.0
    application_kwargs['write_timeout'] = 10.0
    
    # Add important settings for running in a separate thread
    builder = Application.builder().token(BOT_TOKEN)
    
    # Disable signal handlers when running in a non-main thread
    if threading.current_thread() is not threading.main_thread():
        logger.info("Running in non-main thread, disabling signal handlers")
        builder = builder.arbitrary_callback_data(True).concurrent_updates(False)
        # Do not use default signal handlers that cause the error
        application_kwargs['updater'] = None
    else:
        # Enable job queue for main thread
        try:
            from telegram.ext import JobQueue
            logger.info("JobQueue support enabled")
        except ImportError:
            logger.warning("JobQueue not available. Some features will be limited.")
    
    
    app = builder.build()
    
    # Add components to application context
    app.bot_data["user_auth"] = user_auth
    app.bot_data["pico_manager"] = pico_manager
    app.bot_data["controller"] = controller
    app.bot_data["data_manager"] = data_manager
    app.bot_data["sleep_analyzer"] = sleep_analyzer
    
    # Setup handlers
    setup_command_handlers(app)
    setup_message_handlers(app)
    setup_ventilation_handlers(app)
    setup_sleep_handlers(app)
    
    # Start polling with specific settings based on thread
    logger.info("Bot starting polling...")
    if threading.current_thread() is threading.main_thread():
        # In main thread, use the standard polling method
        await app.run_polling()
    else:
        # In non-main thread, we can't use the standard signal handlers
        # Set up polling with non-blocking behavior
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)  
        logger.info("Bot polling running in thread")
        try:
            # Run until the main app stops
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Bot polling cancelled")
        finally:
            await app.stop()
            await app.shutdown()

def main(pico_manager=None, controller=None, data_manager=None, sleep_analyzer=None):
    """Start the bot with proper event loop handling."""
    try:
        logger.info("Starting telegram bot")
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the bot with the appropriate mode based on thread
        if threading.current_thread() is threading.main_thread():
            # In main thread, we can use run_until_complete
            loop.run_until_complete(async_main(pico_manager, controller, data_manager, sleep_analyzer))
        else:
            # In a separate thread, run without blocking
            loop.create_task(async_main(pico_manager, controller, data_manager, sleep_analyzer))
            loop.run_forever()
        
    except Exception as e:
        logger.critical(f"Error starting bot: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Bot stopped by keyboard interrupt")
    finally:
        # Always close the loop to free resources
        try:
            loop.close()
        except:
            pass

if __name__ == "__main__":
    main()