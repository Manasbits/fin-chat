import argparse
import asyncio
import os
from textwrap import dedent
from typing import Optional

from agno.agent import Agent
from agno.models.google import Gemini
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.storage.sqlite import SqliteStorage
from agno.media import Image, Audio, Video  # Add these imports
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Initialize memory and storage
def setup_memory_and_storage():
    """Setup memory and storage for the agent"""
    # Database file for both memory and storage
    db_file = "tmp/tara_agent.db"
    
    # Create directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)
    
    # Initialize memory database for user memories
    memory_db = SqliteMemoryDb(
        table_name="user_memories", 
        db_file=db_file
    )
    
    # Initialize memory with Gemini model for creating memories
    memory = Memory(
        model=Gemini(id="gemini-2.0-flash-lite"),
        db=memory_db
    )
    
    # Initialize storage for session history
    storage = SqliteStorage(
        table_name="agent_sessions", 
        db_file=db_file
    )
    
    return memory, storage

# Setup memory and storage
memory, storage = setup_memory_and_storage()

finance_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-lite"),  # This model supports multimodal
    system_message=dedent("""\
    You are Tara, a friendly financial advisor for young Indians (22-35). 
    
    CORE TRAITS:
    - Warm, approachable, and non-judgmental
    - Speaks in English/Hinglish/hindi (based on user's language preference)
    - Always makes sure to address the user's query
    - Never gives direct investment advice
    - Breaks down complex topics into bite-sized messages
    - Always ends with a follow-up question, for giving better persoanlized answers
    - Remember user preferences and previous conversations to provide personalized advice
    - Can understand and respond to images, audio, and text inputs
    - If user sends an image, analyze it and provide relevant financial advice
    - If user sends audio, transcribe and respond to the content
    
    MESSAGE FORMAT:
    1. First message: Direct answer to user's query (It should be a gem to get the user's attention).
    2. Decide whether it can be answered in 1-2 sentences or it requires more than that.
    3. Second message: Ask a relevant deeper question to continue the conversation, and educate the user about the topic.
    
    [Rest of your system message remains the same...]
    """),
    
    # Memory and Storage Configuration
    memory=memory,
    storage=storage,
    
    # Enable user memories to learn about user preferences
    enable_user_memories=True,
    
    # Enable session summaries for long conversations
    enable_session_summaries=True,
    
    # Add chat history to messages for context
    add_history_to_messages=True,
    num_history_runs=3,
    
    # Enable the agent to read chat history when needed
    read_chat_history=True,
    
    add_datetime_to_instructions=True,
    markdown=True,
)

async def stream_response(message_func, text):
    """Stream response in chunks, breaking at paragraph boundaries."""
    # Split into paragraphs and remove empty ones
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Send each paragraph as a single message
    for para in paragraphs:
        if para:  # Only process non-empty paragraphs
            await message_func(para)
            await asyncio.sleep(0.3)  # Small delay between paragraphs

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming Telegram messages with multimodal support."""
    if not update.message:
        return
    
    user_input = ""
    images = []
    audio = []
    videos = []
    
    # Handle text messages
    if update.message.text:
        user_input = update.message.text
    
    # Handle photo messages
    if update.message.photo:
        # Get the largest photo
        photo = update.message.photo[-1]
        photo_file = await context.bot.get_file(photo.file_id)
        
        # Download photo content
        response = requests.get(photo_file.file_path)
        photo_content = response.content
        
        images.append(Image(content=photo_content))
        
        # If there's a caption, use it as user input
        if update.message.caption:
            user_input = update.message.caption
        else:
            user_input = "Please analyze this image and provide relevant financial advice."
    
    # Handle voice messages
    if update.message.voice:
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        
        # Download voice content
        response = requests.get(voice_file.file_path)
        voice_content = response.content
        
        audio.append(Audio(content=voice_content, format="ogg"))
        user_input = "Please transcribe and respond to this audio message."
    
    # Handle audio files
    if update.message.audio:
        audio_file = await context.bot.get_file(update.message.audio.file_id)
        
        # Download audio content
        response = requests.get(audio_file.file_path)
        audio_content = response.content
        
        audio.append(Audio(content=audio_content))
        user_input = "Please analyze this audio file and provide relevant financial advice."
    
    # Handle video messages
    if update.message.video:
        video_file = await context.bot.get_file(update.message.video.file_id)
        
        # Download video content
        response = requests.get(video_file.file_path)
        video_content = response.content
        
        videos.append(Video(content=video_content))
        
        if update.message.caption:
            user_input = update.message.caption
        else:
            user_input = "Please analyze this video and provide relevant financial advice."
    
    # Handle documents (images as documents)
    if update.message.document:
        document = update.message.document
        if document.mime_type and document.mime_type.startswith('image/'):
            doc_file = await context.bot.get_file(document.file_id)
            
            # Download document content
            response = requests.get(doc_file.file_path)
            doc_content = response.content
            
            images.append(Image(content=doc_content))
            
            if update.message.caption:
                user_input = update.message.caption
            else:
                user_input = "Please analyze this image and provide relevant financial advice."
    
    # If no content was found, return
    if not user_input and not images and not audio and not videos:
        await update.message.reply_text("Sorry, I couldn't process your message. Please send text, image, or audio.")
        return
    
    user_id = str(update.effective_user.id)
    session_id = f"telegram_{user_id}"
    
    # Send a typing action to show the bot is working
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, 
        action='typing'
    )
    
    try:
        # Get the response with multimodal inputs
        response = await asyncio.to_thread(
            finance_agent.run,
            user_input,
            user_id=user_id,
            session_id=session_id,
            images=images if images else None,
            audio=audio if audio else None,
            videos=videos if videos else None,
            stream=False
        )
        
        # Extract the content from the response
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Stream the response in chunks
        await stream_response(
            lambda text: update.message.reply_text(text),
            response_content
        )
        
    except Exception as e:
        await update.message.reply_text(f"Sorry, I encountered an error: {str(e)}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    user_id = str(update.effective_user.id)
    session_id = f"telegram_{user_id}"
    
    # Check if user has previous memories
    user_memories = memory.get_user_memories(user_id=user_id)
    
    if user_memories:
        # Personalized welcome for returning users
        welcome_msg = (
            "Namaste! Main Tara hoon - aapki smart AI-based financial buddy. 💰\n\n"
            "Accha laga aapko phir se dekh kar! Main aapki previous conversations yaad rakhti hoon, "
            "toh aap jahan chhoda tha wahan se continue kar sakte hain.\n\n"
            "Koi naya financial sawaal hai ya previous topic pe baat karni hai? 😊"
        )
    else:
        # Welcome message for new users
        welcome_msg = (
            "Namaste! Main Tara hoon - aapki smart AI-based financial buddy. 💰\n\n"
            "Mujhe pata aap tech-savvy ho, lekin kabhi-kabhi paise manage karna thoda confusing lag sakta hai. "
            "Koi baat nahi, main hoon na aapki madad ke liye!\n\n"
            "Aap mujhe koi bhi financial sawaal pooch sakte hain. Main aapki madad karne ke liye yahaan hoon! 😊"
        )
    
    await update.message.reply_text(welcome_msg)

async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show user what the bot remembers about them."""
    user_id = str(update.effective_user.id)
    
    # Get user memories
    user_memories = memory.get_user_memories(user_id=user_id)
    
    if user_memories:
        memories_text = "Main aapke baare mein yeh yaad rakhti hoon:\n\n"
        for i, mem in enumerate(user_memories[:5], 1):  # Show only first 5 memories
            memories_text += f"{i}. {mem.memory}\n"
        
        if len(user_memories) > 5:
            memories_text += f"\n... aur {len(user_memories) - 5} aur memories bhi hain!"
    else:
        memories_text = "Abhi tak main aapke baare mein kuch specific yaad nahi rakha hai. Thoda aur baat karte hain toh main aapko better samajh paungi! 😊"
    
    await update.message.reply_text(memories_text)

async def clear_memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear user's memories."""
    user_id = str(update.effective_user.id)
    
    # Clear user memories
    memory.delete_user_memory(user_id=user_id)
    
    await update.message.reply_text(
        "Theek hai! Main aapke baare mein sab kuch bhool gayi hoon. "
        "Ab hum fresh start kar sakte hain! 😊"
    )

def run_telegram_bot(token: str) -> None:
    """Run the Telegram bot."""
    print("Starting Telegram bot with multimodal support...")  # Updated message
    
    application = Application.builder().token(token).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("memory", memory_command))
    application.add_handler(CommandHandler("clear_memory", clear_memory_command))
    
    # THIS IS THE KEY FIX - Replace the old handler with multimodal support
    application.add_handler(MessageHandler(
        filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO | filters.VIDEO | filters.Document.ALL,
        handle_message
    ))
    
    print("Bot is running with multimodal support. Press Ctrl+C to stop.")
    print("Supported inputs:")
    print("- Text messages")
    print("- Photos/Images") 
    print("- Voice messages")
    print("- Audio files")
    print("- Video files")
    print("Available commands:")
    print("- /start - Start conversation")
    print("- /memory - See what the bot remembers about you")
    print("- /clear_memory - Clear your memory")
    application.run_polling()
    
async def run_terminal() -> None:
    """Run the agent in terminal mode."""
    print("\n" + "="*50)
    print("Namaste! Main Tara hoon - aapki smart AI-based financial buddy.")
    print("Terminal mode with memory enabled!")
    
    # Use a default user ID for terminal mode
    user_id = "terminal_user"
    session_id = "terminal_session"
    
    async def print_streamed(text):
        print("\nTara:", end=" ")
        for chunk in text.split('\n'):
            if chunk.strip():
                print(chunk.strip())
                await asyncio.sleep(0.3)
        print()  # Add an extra newline at the end
    
    while True:
        try:
            user_input = input("\nAap: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye', 'alvida']:
                print("\nAlvida! Phir milenge. 😊")
                break
            
            # Special commands for terminal
            if user_input.lower() == '/memory':
                user_memories = memory.get_user_memories(user_id=user_id)
                if user_memories:
                    print("\nMain aapke baare mein yeh yaad rakhti hoon:")
                    for i, mem in enumerate(user_memories, 1):
                        print(f"{i}. {mem.memory}")
                else:
                    print("\nAbhi tak koi specific memories nahi hain.")
                continue
            
            if user_input.lower() == '/clear_memory':
                memory.delete_user_memory(user_id=user_id)
                print("\nMemory clear kar di! Fresh start! 😊")
                continue
                
            if not user_input:
                continue
                
            # Run the blocking call in a separate thread
            response = await asyncio.to_thread(
                finance_agent.run,
                user_input,
                user_id=user_id,
                session_id=session_id,
                stream=False
            )
            
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Print a typing indicator
            print("\nTara likh rahi hoon...\n")
            
            # Stream the response
            await print_streamed(response_content)
            
        except KeyboardInterrupt:
            print("\n\nAlvida! Phir milenge. 😊")
            break
        except Exception as e:
            print(f"\nMaaf, kuch to gadbad hai: {str(e)}")

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Tara - Your Financial Assistant with Memory')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--terminal', action='store_true', help='Run in terminal mode')
    group.add_argument('--telegram', action='store_true', help='Run Telegram bot using token from .env')
    
    args = parser.parse_args()
    
    if args.telegram:
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not telegram_token:
            print("Error: TELEGRAM_BOT_TOKEN not found in environment variables")
            print("Please make sure you have TELEGRAM_BOT_TOKEN=your_token_here in your .env file")
            return
        run_telegram_bot(telegram_token)
    else:
        asyncio.run(run_terminal())

if __name__ == "__main__":
    main()