# bot/services.py
"""Background services for the bot."""
import asyncio
import logging
from telegram import Bot
from utils.network_scanner import scan_network
from presence.device_manager import DeviceManager

logger = logging.getLogger(__name__)

async def telegram_ping_worker(bot: Bot, device_manager: DeviceManager, telegram_ping_queue: asyncio.Queue):
    """
    Worker function that processes Telegram ping requests.
    
    Args:
        bot: Telegram bot instance
        device_manager: DeviceManager instance
        telegram_ping_queue: Queue with ping tasks
    """
    logger.info("Starting Telegram ping worker")
    
    while True:
        try:
            # Get next ping task from queue
            task_data = await asyncio.get_event_loop().run_in_executor(None, telegram_ping_queue.get)
            
            mac = task_data['mac']
            telegram_user_id = task_data['telegram_user_id']
            ip_address = task_data['ip_address']
            
            logger.info(f"Processing Telegram ping for device {mac} (user {telegram_user_id})")
            
            # Send silent message
            try:
                sent_message = await bot.send_message(
                    chat_id=telegram_user_id, 
                    text=".", 
                    disable_notification=True
                )
                logger.info(f"Sent Telegram ping to user {telegram_user_id} for device {mac}")
                
                # Delete message after a brief pause
                await asyncio.sleep(0.5)
                try:
                    await bot.delete_message(
                        chat_id=sent_message.chat_id, 
                        message_id=sent_message.message_id
                    )
                    logger.debug(f"Deleted ping message {sent_message.message_id} for user {telegram_user_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete ping message: {e}")
                
            except Exception as e:
                logger.error(f"Failed to send Telegram ping to user {telegram_user_id}: {e}")
                # Signal failure
                device_manager.process_telegram_ping_result(mac, False)
                telegram_ping_queue.task_done()
                continue
            
            # Wait for device to wake up
            await asyncio.sleep(10)
            
            # Scan for device
            try:
                scan_results = scan_network(target_ip=ip_address)
                detected = any(entry[0].lower() == mac.lower() for entry in scan_results)
                
                logger.debug(f"Post-ping scan result for {mac}: detected={detected}")
                device_manager.process_telegram_ping_result(mac, detected)
                
            except Exception as e:
                logger.error(f"Error during post-ping scan for {mac}: {e}")
                device_manager.process_telegram_ping_result(mac, False)
            
            # Mark task as done
            telegram_ping_queue.task_done()
            
        except asyncio.CancelledError:
            logger.info("Telegram ping worker cancelled")
            break
        except Exception as e:
            logger.error(f"Error in Telegram ping worker: {e}")
            await asyncio.sleep(1)
    
    logger.info("Telegram ping worker stopped")