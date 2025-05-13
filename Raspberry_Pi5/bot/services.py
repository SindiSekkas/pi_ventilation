# bot/services.py
"""Background services for the bot."""
import asyncio
import logging
from telegram import Bot
from utils.network_scanner import scan_network # Убедитесь, что scan_network импортирован
from presence.device_manager import DeviceManager

logger = logging.getLogger(__name__)

async def telegram_ping_worker(bot: Bot, device_manager: DeviceManager, telegram_ping_queue: asyncio.Queue):
    """
    Asynchronously processes device ping requests received via a queue.

    This worker attempts to "wake" a target device by sending a silent Telegram
    notification to the associated user. After a delay (to allow the device to
    connect to the network if woken by the notification), it performs a network
    scan to check for the device's presence. The result (detected or not) is
    then reported back to the DeviceManager.

    Args:
        bot (Bot): The Telegram bot instance used for sending notifications.
        device_manager (DeviceManager): The manager responsible for device state
                                       and processing ping results.
        telegram_ping_queue (asyncio.Queue): The queue from which ping tasks are retrieved.
                                             Each task is a dictionary expecting 'mac',
                                             'telegram_user_id', and optionally 'ip_address'.
    """
    logger.info("Starting Telegram ping worker")
    
    while True:
        try:
            # Retrieve the next ping task. Using run_in_executor for the blocking queue.get()
            # to avoid halting the asyncio event loop.
            task_data = await asyncio.get_event_loop().run_in_executor(None, telegram_ping_queue.get)
            
            mac = task_data['mac']
            telegram_user_id = task_data['telegram_user_id']
            ip_address = task_data.get('ip_address') # IP address is optional for targeted scanning.
            
            logger.info(f"Processing Telegram ping for device {mac} (user {telegram_user_id}, IP: {ip_address})")
            
            # Attempt to wake the device by sending a silent Telegram notification.
            # This might trigger the device to come online if it's configured to react to notifications.
            try:
                sent_message = await bot.send_message(
                    chat_id=telegram_user_id, 
                    text=".", # A minimal, non-intrusive message.
                    disable_notification=True # Ensures the user isn't actively disturbed.
                )
                logger.info(f"Sent Telegram ping to user {telegram_user_id} for device {mac}")
                
                # Clean up the ping message shortly after sending.
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
                # Report failure to DeviceManager if the ping message couldn't be sent.
                # Ensure device_manager is not None before calling
                if device_manager:
                    device_manager.process_telegram_ping_result(mac, False)
                telegram_ping_queue.task_done() # Ensure the task is marked as completed in the queue.
                continue # Proceed to the next task.
            
            # Allow some time for the device to wake up and connect to the network
            # after receiving the Telegram notification.
            await asyncio.sleep(15) 
            
            # Perform a network scan to check if the device is now present.
            try:
                detected_after_ping = False
                loop = asyncio.get_event_loop() # Получаем текущий event loop
                
                scan_results = await loop.run_in_executor(None, scan_network, ip_address)
                
                for device_mac, device_ip_found, _ in scan_results:
                    if device_mac.lower() == mac.lower():
                        detected_after_ping = True
                        logger.info(f"Device {mac} detected on network after Telegram ping (IP: {device_ip_found})")
                        break # Device found.
                
                logger.debug(f"Post-ping scan result for {mac}: detected={detected_after_ping}")
                # Report the outcome of the ping and scan attempt to the DeviceManager.
                if device_manager:
                    device_manager.process_telegram_ping_result(mac, detected_after_ping)
                
            except Exception as e:
                logger.error(f"Error during post-ping scan for {mac}: {e}")
                # Report failure if the scan itself encounters an error.
                if device_manager:
                    device_manager.process_telegram_ping_result(mac, False)
            
            # Indicate that processing for this task is complete.
            telegram_ping_queue.task_done()
            
        except asyncio.CancelledError:
            logger.info("Telegram ping worker cancelled")
            break # Exit the loop if the worker task itself is cancelled.
        except Exception as e:
            logger.error(f"Unhandled error in Telegram ping worker loop: {e}", exc_info=True)
            # Brief pause to prevent rapid looping in case of persistent errors not caught above.
            await asyncio.sleep(1) 
    
    logger.info("Telegram ping worker stopped")