import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import time
import traceback
from datetime import datetime
from textwrap import dedent
from typing import Optional, Dict, Any, List, Union

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.google import Gemini
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.storage.sqlite import SqliteStorage
from agno.media import Image, Audio, Video
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
from io import BytesIO
import uvicorn

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
        try:
            # Get the response with multimodal inputs
            response = await finance_agent.astream(
                message,
                images=images,
                audio=audio,
                videos=videos,
                user_id=user_id,
                session_id=session_id,
            )
            
            # Stream the response back to WhatsApp
            full_response = ""
            chunk_buffer = ""
            
            async for chunk in response:
                full_response += chunk
                chunk_buffer += chunk
                
                # Send in chunks of reasonable size
                if len(chunk_buffer) > 100 or '\n' in chunk_buffer:
                    await send_whatsapp_message(phone_number, chunk_buffer)
                    chunk_buffer = ""
                    # Show typing indicator between chunks
                    await send_whatsapp_typing(phone_number, True)
            
            # Send any remaining content
            if chunk_buffer:
                await send_whatsapp_message(phone_number, chunk_buffer)
                
            # Update conversation history in memory
            if 'conversation' not in user_memories[user_id]:
                user_memories[user_id]['conversation'] = []
                
            user_memories[user_id]['conversation'].append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.now().isoformat(),
                'media_type': media_type
            })
            
            user_memories[user_id]['conversation'].append({
                'role': 'assistant',
                'content': full_response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep conversation history manageable
            if len(user_memories[user_id]['conversation']) > 20:
                user_memories[user_id]['conversation'] = user_memories[user_id]['conversation'][-20:]
                
            return full_response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            print("Response generation error:", traceback.format_exc())
            await send_whatsapp_message(phone_number, "I encountered an error processing your request. Please try again.")
            return ""
            
        finally:
            # Always ensure typing indicator is turned off
            await send_whatsapp_typing(phone_number, False)
            
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Stream the response in chunks
        await stream_response(
            lambda text: update.message.reply_text(text),
{{ ... }}

class WhatsAppWebhookPayload(BaseModel):
    object: str
    entry: List[Dict[str, Any]]

async def send_whatsapp_typing(phone_number: str, is_typing: bool) -> bool:
    """Send typing indicator to WhatsApp."""
    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    action = "typing_on" if is_typing else "typing_off"
    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": phone_number,
        "type": "text",
        "text": {"preview_url": False, "body": ""},
        "status": "sent",
        "messaging_type": "RESPONSE",
        "recipient_name": "User",
        "sender": {"name": "Tara"},
        "action": {"type": action}
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    error = await response.text()
                    print(f"Error sending typing indicator: {error}")
                return response.status == 200
    except Exception as e:
        print(f"Exception sending typing indicator: {e}")
        return False

async def send_whatsapp_message(phone_number: str, message: str, reply_to: Optional[str] = None) -> bool:
    """Send a message via WhatsApp with support for replies."""
    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Basic message data
    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": phone_number,
        "type": "text",
        "text": {"preview_url": True, "body": message}
    }
    
    # Add context for replies if available
    if reply_to:
        data["context"] = {"message_id": reply_to}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    error = await response.text()
                    print(f"Error sending WhatsApp message: {error}")
                    return False
                return True
    except Exception as e:
        print(f"Exception sending WhatsApp message: {e}")
        return False

async def process_whatsapp_message(phone_number: str, message: str, media_type: str = None, media_id: str = None):
    """Process incoming WhatsApp messages and generate responses with memory and typing indicators."""
    user_id = f"whatsapp_{phone_number}"
{{ ... }}
    session_id = f"whatsapp_{phone_number}_{int(time.time())}"
    
    # Initialize memory if not exists
    if user_id not in user_memories:
        user_memories[user_id] = {}
    
    # Get or create user session
    if 'session' not in user_memories[user_id]:
        user_memories[user_id]['session'] = {
            'start_time': datetime.now().isoformat(),
            'message_count': 0,
            'last_active': datetime.now().isoformat()
        }
    
    # Update session info
    user_memories[user_id]['session']['message_count'] += 1
    user_memories[user_id]['session']['last_active'] = datetime.now().isoformat()
    
    # Send typing indicator
    await send_whatsapp_typing(phone_number, True)
    
    # Prepare media inputs
    images, audio, videos = [], [], []
    
    if media_type and media_id:
        try:
            # First, get the media URL using the media ID
            media_url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{media_id}"
            headers = {
                "Authorization": f"Bearer {WHATSAPP_TOKEN}",
                "Content-Type": "application/json"
            }
            
            # Get the actual media URL
            response = requests.get(media_url, headers=headers)
            response.raise_for_status()
            media_data = response.json()
            
            if 'url' not in media_data:
                print(f"No URL found in media data: {media_data}")
                await send_whatsapp_message(phone_number, "Sorry, I couldn't process the media. Please try again.")
                return
                
            # Download the actual media content
            download_url = media_data['url']
            media_response = requests.get(download_url, headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"})
            media_response.raise_for_status()
            content = media_response.content
            
            # Process based on media type
            if media_type == 'image':
                images.append(Image(content=content))
                if not message:
                    message = "Please analyze this image and provide relevant financial advice."
            elif media_type == 'audio':
                audio.append(Audio(content=content))
                if not message:
                    message = "Please transcribe and respond to this audio message."
            elif media_type == 'video':
                videos.append(Video(content=content))
                if not message:
                    message = "Please analyze this video and provide relevant financial advice."
            elif media_type == 'document':
                # For documents, we'll try to handle images
                images.append(Image(content=content))
                if not message:
                    message = "Please analyze this document and provide relevant financial advice."
                    
        except requests.exceptions.RequestException as e:
            error_msg = f"Error downloading media: {str(e)}"
            print(error_msg)
            await send_whatsapp_message(phone_number, "Sorry, I encountered an error processing the media. Please try again.")
            return
        except Exception as e:
            error_msg = f"Error processing media: {str(e)}"
            print(error_msg)
            print("Media processing error:", traceback.format_exc())
            await send_whatsapp_message(phone_number, "Sorry, I couldn't process the media. Please try again with a different file.")
            return
    
    try:
        # Get the response with multimodal inputs
        response = await asyncio.to_thread(
            finance_agent.run,
            message or "Hello!",
            user_id=user_id,
            session_id=session_id,
            images=images if images else None,
            audio=audio if audio else None,
            videos=videos if videos else None,
            stream=False
        )
        
        # Extract the content from the response
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Send the response back to WhatsApp
        await send_whatsapp_message(phone_number, response_content)
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        await send_whatsapp_message(phone_number, error_msg)

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Verify webhook for WhatsApp API."""
    query_params = request.query_params
    mode = query_params.get("hub.mode")
    token = query_params.get("hub.verify_token")
    challenge = query_params.get("hub.challenge")
    
    print(f"Verification request - Mode: {mode}, Token: {token}, Challenge: {challenge}")
    
    if mode and token:
        if mode == 'subscribe' and token == WHATSAPP_VERIFY_TOKEN:
            print("Webhook verified successfully")
            return Response(content=challenge, media_type="text/plain")
    
    print("Webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming WhatsApp messages via webhook."""
    try:
        # Parse the request body
        try:
            data = await request.json()
            print("Incoming webhook data:", json.dumps(data, indent=2))
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return JSONResponse(content={"status": "error", "message": "Invalid JSON"}, status_code=400)
        
        # Check if this is a WhatsApp API event
        if 'object' not in data or 'entry' not in data:
            print("Invalid webhook format - missing 'object' or 'entry'")
            return JSONResponse(content={"status": "ignored"}, status_code=200)
        
        # Process each entry
        for entry in data.get('entry', []):
            try:
                for change in entry.get('changes', []):
                    value = change.get('value', {})
                    
                    # Check if this is a message
                    messages = value.get('messages', [])
                    if not messages:
                        continue
                        
                    for message in messages:
                        try:
                            if not all(key in message for key in ['from', 'id']):
                                print("Invalid message format - missing required fields")
                                continue
                                
                            phone_number = message['from']
                            message_id = message['id']
                            print(f"Processing message {message_id} from {phone_number}")
                            
                            # Handle different message types
                            if 'text' in message:
                                text = message['text'].get('body', '')
                                if not text.strip():
                                    print("Empty text message received")
                                    continue
                                print(f"Processing text message: {text[:100]}...")
                                asyncio.create_task(process_whatsapp_message(phone_number, text))
                                
                            elif 'image' in message:
                                image = message.get('image', {})
                                if 'id' not in image:
                                    print("Image message missing ID")
                                    continue
                                image_id = image['id']
                                caption = image.get('caption', '')
                                print(f"Processing image message with ID: {image_id}")
                                asyncio.create_task(process_whatsapp_message(phone_number, caption, 'image', image_id))
                                
                            elif 'audio' in message:
                                audio = message.get('audio', {})
                                if 'id' not in audio:
                                    print("Audio message missing ID")
                                    continue
                                audio_id = audio['id']
                                print(f"Processing audio message with ID: {audio_id}")
                                asyncio.create_task(process_whatsapp_message(phone_number, "", 'audio', audio_id))
                                
                            elif 'video' in message:
                                video = message.get('video', {})
                                if 'id' not in video:
                                    print("Video message missing ID")
                                    continue
                                video_id = video['id']
                                print(f"Processing video message with ID: {video_id}")
                                asyncio.create_task(process_whatsapp_message(phone_number, "", 'video', video_id))
                                
                            elif 'document' in message:
                                document = message.get('document', {})
                                if 'id' not in document:
                                    print("Document message missing ID")
                                    continue
                                doc_id = document['id']
                                print(f"Processing document with ID: {doc_id}")
                                asyncio.create_task(process_whatsapp_message(phone_number, "", 'document', doc_id))
                                
                        except Exception as e:
                            print(f"Error processing individual message: {e}")
                            continue
                            
            except Exception as e:
                print(f"Error processing entry: {e}")
                continue
        
        return JSONResponse(content={"status": "success"}, status_code=200)
    
    except Exception as e:
        error_msg = f"Unexpected error in webhook: {e}"
        print(error_msg)
        print("Full traceback:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

def run_whatsapp_webhook(host: str = "0.0.0.0", port: int = 8000):
    """Run the WhatsApp webhook server."""
    print(f"Starting WhatsApp webhook server on http://{host}:{port}")
    print(f"Webhook URL: https://your-domain.com/webhook")
    print(f"Verification Token: {WHATSAPP_VERIFY_TOKEN}")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(app, host=host, port=port)

async def run_services(telegram: bool = False, whatsapp: bool = False, host: str = '0.0.0.0', port: int = 8000):
    """Run the specified services concurrently."""
    tasks = []
    
    if telegram:
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not token:
            print("Error: TELEGRAM_BOT_TOKEN environment variable not set")
            return
        print("Starting Telegram bot...")
        # Create a task for the Telegram bot
        telegram_task = asyncio.create_task(asyncio.to_thread(run_telegram_bot, token))
        tasks.append(telegram_task)
    
    if whatsapp:
        if not all([os.getenv('WHATSAPP_TOKEN'), os.getenv('WHATSAPP_PHONE_NUMBER_ID')]):
            print("Error: WhatsApp environment variables not properly configured")
            return
        print(f"Starting WhatsApp webhook server on {host}:{port}...")
        # Create a task for the WhatsApp webhook server
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            workers=1
        )
        server = uvicorn.Server(config)
        whatsapp_task = asyncio.create_task(server.serve())
        tasks.append(whatsapp_task)
    
    if not tasks:
        print("No services specified to run")
        return
    
    # Run all tasks concurrently
    await asyncio.gather(*tasks)

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Tara - Your Financial Assistant with Memory')
    
    # Make the group non-exclusive to allow multiple services
    group = parser.add_argument_group('Services')
    group.add_argument('--terminal', action='store_true', help='Run in terminal mode')
    group.add_argument('--telegram', action='store_true', help='Run Telegram bot using token from .env')
    group.add_argument('--whatsapp', action='store_true', help='Run WhatsApp webhook server')
    
    # Add WhatsApp webhook server options
    whatsapp_group = parser.add_argument_group('WhatsApp Webhook Options')
    whatsapp_group.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the webhook server on')
    whatsapp_group.add_argument('--port', type=int, default=8000, help='Port to run the webhook server on')
    
    args = parser.parse_args()
    
    # Check if no service is specified
    if not any([args.terminal, args.telegram, args.whatsapp]):
        parser.print_help()
        return
    
    if args.terminal:
        asyncio.run(run_terminal())
    else:
        # Check if Telegram is requested but token is missing
        if args.telegram and not os.getenv('TELEGRAM_BOT_TOKEN'):
            print("Error: TELEGRAM_BOT_TOKEN not found in .env file")
            return
            
        asyncio.run(run_services(
            telegram=args.telegram,
            whatsapp=args.whatsapp,
            host=args.host,
            port=args.port
        ))

if __name__ == "__main__":
    main()