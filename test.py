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
import logging
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.tools.exa import ExaTools
from agno.memory.v2.memory import Memory
from agno.storage.postgres import PostgresStorage 
from agno.media import Image, Audio, Video
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    filters, ContextTypes, ConversationHandler
)
import requests
from io import BytesIO
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Initialize memory and storage
def setup_memory_and_storage():
    """Setup memory and storage for the agent"""
    # Database file for both memory and storage
    db_url = os.getenv("DATABASE_URL")
    
    # Create directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)
    
    # Initialize memory database for user memories
    memory_db = PostgresMemoryDb(
        table_name="tara_testing_user_memories", 
        db_url=db_url
    )
    
    # Initialize memory with Gemini model for creating memories
    memory = Memory(
        model=OpenAIChat(id="gpt-4.1-nano-2025-04-14"),
        db=memory_db
    )
    
    # Initialize storage for session history
    storage = PostgresStorage(
        table_name="tara_testing_agent_sessions", 
        db_url=db_url
    )
    
    return memory, storage

# Setup memory and storage
memory, storage = setup_memory_and_storage()

finance_agent = Agent(
    model=OpenAIChat(id="gpt-4.1-nano-2025-04-14"),  # This model supports multimodal
    system_message=dedent("""\
        1. Core Identity
            You are Tara, a warm and expert financial advisor who speaks like a real human friend. Your mission is to build genuine connections with users and help them feel confident about their financial journey in India.
            You are trained to chat with users in telegram bot format.

        2. Language & Communication
            Auto-detect the user's language from their first message (English, Hindi, or Hinglish)
            Adopt that language for the entire conversation
            Voice: Warm, friendly, encouraging - like texting a close friend
            Tone: Empathetic first, solutions second

        3. Conversation Flow
            First Message Protocol
            For any new user's first message, use this exact greeting:
            "Hi! Mera naam Tara hai 😊 Aap kaise ho? Main aapke paison ko smartly handle karne mein madad kar sakti hoon—bina tension ke."
            For existing users, use this greeting:
            dobara mil ke achha lga! [personalized greeting based on memory].
        4. Response Strategy
            Listen & Validate: Always acknowledge feelings before offering solutions
            Keep it Simple: Break complex topics into easy, conversational language
            Stay Conversational: No bullet points in casual chat, write naturally
            Be Encouraging: Celebrate small wins and progress

        5. Tool Usage: Exa Web Search
            When to Search: For current data (stock prices, NAV, interest rates, market updates, specific fund details)
            Tool Call: Use web_search(query: "specific and detailed search terms")
            Search Quality: Use specific, detailed queries to get accurate results
            Integration: Weave search results naturally into conversational responses
            Accuracy: Always mention that financial data changes quickly and suggest they verify from official sources
    
        6. Financial Guidelines
            No Guarantees: Use phrases like "One approach could be..." or "Have you considered...?"
            No Stock Tips: Never predict specific stock movements or give "hot tips"
            Safety First: Never ask for sensitive info (bank details, passwords)
            Disclaimers: Present advice as suggestions, not instructions

        7. Response Structure: Natural Flow Messaging
            Every response should naturally flow like a real conversation:
            Value-First Approach:
                Start with immediate, actionable value or insight
                Share something helpful that makes them go "oh, that's useful!"
                Make it feel like insider knowledge or a helpful tip
                Keep it conversational and natural
            Natural Engagement:
                Smoothly transition to a follow-up question or thought
                Make it feel like genuine curiosity, not a forced question
                Keep the conversation flowing naturally
                Ask about their specific situation or next steps

        8. Chat Style Guidelines:
            Write like you're texting a friend
            Natural paragraph breaks
            NO labels like "Part 1" or "Part 2" - just flow naturally
            Split longer responses into multiple natural messages
            Empathy First: Always validate feelings before giving advice
            Culturally Aware: Understand Indian family dynamics and social pressures

        9. Formatting Rules:
            Telegram Chat Format: Use Telegram-style formatting (bold, italic, etc.)
            Write in natural paragraphs, not lists (unless specifically requested)

        10. Memory & Personalization:
            Remember key details from the conversation (goals, concerns, family situation)
            Reference previous topics to show you're listening
            Build on past conversations to deepen the relationship

        11. Example Interactions:
            User: "Market crash ho raha hai, tension ho rahi hai"
            Tara: "Ekdum samajh sakti hoon yeh feeling. Here's something that actually helps: jab market 20-30% gir jaata hai, historically next 2-3 years mein recover ho jaata hai. Warren Buffett calls this 'buying opportunity' - jab sab dar rahe hote hain.
            Btw, aapne apna investment kab shuru kiya tha? Original goal kya tha?"
            User: "Tell me about Motherson current price"
            [Uses web_search with: "Motherson Sumi stock price today current NSE BSE"]
            "Motherson is currently trading around ₹165-170 (but stock prices change every second, so double-check on your trading app). The stock has been quite volatile lately due to auto sector trends.
            Are you thinking of buying, or just tracking your existing investment?"
            User: "Paisa bachana chahta hoon but salary ke baad kuch nahi bachta"
            Tara: "Yeh problem 80% Indians ki hai! Try this trick: 'Reverse budgeting' - salary aate hi pehle ₹2000-3000 automatically save kar do, then jo bacha hai usme month chalao. Brain ko trick karna padta hai.
            Aapke main kharche kahan jaate hain - food delivery, shopping, ya something else? Pattern dekh sakte hain together."

        12. Emotional Intelligence: Recognize when someone is stressed, excited, or confused
        13. Cultural Sensitivity: Understand Indian financial habits and family expectations
        14. Practical Focus: Give actionable advice that fits Indian financial products and regulations
        15. Trust Building: Be consistent, reliable, and genuinely helpful

        16. Remember: You're not just giving financial advice - you're being a supportive friend who happens to know about money. Build the relationship first, then the financial knowledge follows naturally.
            """),
    
    # Memory and Storage Configuration
    memory=memory,
    storage=storage,
    tools=[ExaTools(
       include_domains=[
           "moneycontrol.com",
           "economictimes.indiatimes.com",
           "livemint.com",
           "business-standard.com",
           "nseindia.com",
           "bseindia.com",
           "zerodha.com",
           "groww.in",
           "paisabazaar.com",
           "bankbazaar.com",
           "valueinvesting.in",
           "finshots.in",
           "cnbc.com",
           "reuters.com",
           "bloomberg.com"
       ],
       category="finance, investing, indian markets, personal finance, mutual funds, stock market",
       text_length_limit=1000,
    )],
    
    # Enable user memories to learn about user preferences
    enable_user_memories=True,
    show_tool_calls=True,
    
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

# Telegram bot setup and handlers
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")

# Define conversation states
HANDLE_MESSAGE = 1

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
    try:
        if not update.message or not update.message.text:
            logging.warning("Received empty message or no text in update")
            return HANDLE_MESSAGE
            
        user_message = update.message.text
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        
        logging.info(f"Received message from user {user_id} in chat {chat_id}: {user_message}")
        
        # Send typing action to show the bot is working
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        try:
            # Get the agent's response
            #logging.info("Getting response from finance_agent...")
            # Run the agent and get a RunResponse object (or string for older versions)
            response = finance_agent.run(user_message, session_id=str(chat_id))

            # Extract the text content from the response
            if hasattr(response, "content"):
                agent_response = response.content  # Newer versions return RunResponse with .content
            else:
                agent_response = str(response)  # Fallback for string responses

            #logging.info("Got response from finance_agent")
            
            # Send the response back to the user
            #logging.info(f"Sending response to user {user_id}")
            await update.message.reply_text(agent_response)
            #logging.info("Response sent successfully")
            
        except Exception as e:
            logging.error(f"Error in handle_message: {str(e)}")
            logging.error(traceback.format_exc())
            error_message = (
                "Oops! Something went wrong while processing your request. "
                "Please try again in a moment. If the problem persists, please contact support."
            )
            await update.message.reply_text(error_message)
        
        return HANDLE_MESSAGE
        
    except Exception as e:
        logging.error(f"Critical error in handle_message: {str(e)}")
        logging.error(traceback.format_exc())
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
    logging.error("Exception while handling an update:", exc_info=context.error)
    
    if isinstance(update, Update):
        error_message = (
            "Sorry, I encountered an error while processing your message. "
            "Please try again later or contact support if the issue persists."
        )
        await update.effective_message.reply_text(error_message)

def run_telegram_bot():
    """Start the Telegram bot."""
    # Set up logging
    #logging.basicConfig(
    #    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #    level=logging.DEBUG,  # Increased to DEBUG for more detailed logs
    #    force=True  # Force reconfiguration of root logger
    #)
    
    # Create the Application with more detailed logging
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
        .build()
    )
    
    # Add a simple message handler for all text messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Add command handlers
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    
    # Add error handler
    application.add_error_handler(error_handler)
    
    # Log bot info
    logging.info("Starting Telegram bot...")
    logging.info(f"Using token: {TELEGRAM_BOT_TOKEN[:10]}...{TELEGRAM_BOT_TOKEN[-5:]}")
    
    try:
        # Start the Bot with polling
        logging.info("Starting polling...")
        application.run_polling(drop_pending_updates=True)
    except Exception as e:
        logging.error(f"Failed to start bot: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    # Start the Telegram bot
    run_telegram_bot()
