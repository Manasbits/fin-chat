
import os
import logging
import traceback
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    filters, ContextTypes, ConversationHandler
)
from test import finance_agent  # Import from test.py where finance_agent is defined
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Telegram bot setup
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")

# Define conversation states (if needed in the future)
MENU, HANDLE_MESSAGE = range(2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Send a welcome message when the command /start is issued."""
    user = update.effective_user
    welcome_message = (
        f"Hi {user.first_name}! 👋\n\n"
        "I'm Tara, your friendly financial assistant. I can help you with:\n"
        "• Stock market information\n"
        "• Investment advice\n"
        "• Personal finance tips\n\n"
        "Just ask me anything about finance! 💰"
    )
    
    await update.message.reply_text(welcome_message)
    return HANDLE_MESSAGE

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle incoming messages and respond using the finance_agent."""
    user_message = update.message.text
    chat_id = update.effective_chat.id
    
    # Send typing action to show the bot is working
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    
    try:
        # Get the agent's response
        agent_response = finance_agent.get_response(user_message, session_id=str(chat_id))
        
        # Send the response back to the user
        await update.message.reply_text(agent_response)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        logger.error(traceback.format_exc())
        error_message = (
            "Oops! Something went wrong while processing your request. "
            "Please try again in a moment. If the problem persists, please contact support."
        )
        await update.message.reply_text(error_message)
    
    return HANDLE_MESSAGE

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        "🤖 *Tara - Your Financial Assistant*\n\n"
        "Here's what I can help you with:\n"
        "• Get stock market information 📈\n"
        "• Investment advice 💰\n"
        "• Personal finance tips 💡\n"
        "• Market analysis 📊\n\n"
        "Just type your question, and I'll do my best to help!"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors and send a message to the user."""
    logger.error("Exception while handling an update:", exc_info=context.error)
    
    if isinstance(update, Update):
        error_message = (
            "Sorry, I encountered an error while processing your message. "
            "Please try again later or contact support if the issue persists."
        )
        await update.effective_message.reply_text(error_message)

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            HANDLE_MESSAGE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message),
                CommandHandler('help', help_command)
            ],
        },
        fallbacks=[CommandHandler('help', help_command)],
    )
    
    application.add_handler(conv_handler)
    
    # Add error handler
    application.add_error_handler(error_handler)
    
    # Start the Bot
    logger.info("Starting bot...")
    application.run_polling()

if __name__ == '__main__':
    main()
