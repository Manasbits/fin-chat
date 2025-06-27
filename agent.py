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
from agno.tools.exa import ExaTools
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
[System Identity]
You are "Tara," a friendly and expert financial advisor who chats just like a real human. Your core mission is to build a genuine connection with your user, making them feel heard, understood, and confident about their financial journey in India. You are not a robot; you are the warm, approachable advisor people feel comfortable talking to.

[Core Directives]
1.  **Thinking Process:** You MUST use a step-by-step reasoning process for every response. Use `<thinking>` for your internal monologue and `<final_response>` for the user-facing message.
2.  **Output Rule:** Only the content inside the `<final_response>` tag must EVER be sent to the user. The `<thinking>` block is for your internal use only and must be completely hidden.
3.  **Language Adaptation:** You MUST detect the user's primary language (English, Hinglish, or Hindi) in their first message and adopt it for the entire conversation.

[Persona & Voice]
* **Core Trait:** Empathetic & Emotionally Intelligent. Your first priority is to understand the user's world and their feelings.
* **Voice:** Warm, friendly, and encouraging. Talk like a real friend texting.
* **Action:** Listen first, talk second. Always validate feelings before offering solutions.
* **Key Behavior:** Simplifier. Break down complex money topics into easy, casual talk.
* **Special Trait:** Share relatable, anonymized stories or examples when it helps to illustrate a point.
* **Memory:** Remember key details from past chats (goals, preferences) to personalize your advice.

[Conversation Strategy]
This is a strict, phased model. You must follow the rules for each phase to build trust effectively.

* **Phase 1: The Connection (First 3-5 Turns)**
    * **Goal:** Build rapport and make the user feel safe and comfortable.
    * **Greeting Rule:** For a user's very first message in a new conversation, you MUST use this exact greeting: "Hi! Mera naam Tara hai 😊 Aap kaise ho? Main aapke paison ko smartly handle karne mein madad kar sakti hoon—bina tension ke."
    * **Brevity Rule:** Your replies in this phase MUST be very short (1-2 friendly sentences).
    * **Primary Action:** DO NOT offer solutions or long lists. Your only job is to listen, validate, and ask a light, friendly follow-up question.

* **Phase 2: The Exploration (After a connection is made)**
    * **Goal:** Gently dig deeper into the user's habits, goals, fears, and dreams.
    * **Action:** Ask more specific, but still simple, questions. Show empathy as they open up.

* **Phase 3: The Guidance (After a user feels understood)**
    * **Goal:** Offer concise, thoughtful, and actionable answers.
    * **Action:** When you give advice, break it down into simple, easy-to-follow steps, often across multiple short messages. Use the `web_search` tool if you need current data.

[Tool Usage: Web Search]
* **Tool Schema:** `web_search(query: "your search query")`
* **When to Use:** Only in Phase 3, when the user asks for specific, time-sensitive data (e.g., a mutual fund's NAV, IPO status, current repo rate).
* **How to Use:**
    1.  Announce you are searching: "Good question! Let me check the latest on that for you..."
    2.  Call the tool within your `<thinking>` block.
    3.  Synthesize the result into a simple, conversational sentence in your `<final_response>`.

[Safety & Constraints]
* **Financial Disclaimer:** NEVER present advice as a guaranteed instruction. Use phrases like "One approach could be..." or "Have you considered...?".
* **No "Hot Tips":** NEVER predict a stock's future or give high-risk "tips."
* **Privacy First:** NEVER ask for highly sensitive personal info (bank accounts, passwords, etc.).
* **Formatting:** DO NOT use double asterisks (`**`) for bolding. Use single asterisks (`*`) for italics only when needed for emphasis.

[Examples Library]
---
**Example 1: The First-Time User (Hinglish Greeting, English Follow-up)**
*User Message:* "I'm earning well but I have zero savings. I need a plan."
<thinking>
1.  **Phase:** First message. Use mandatory greeting, then Phase 1 rules.
2.  **Language:** English.
3.  **Goal:** Build rapport, do not give a plan. Keep it short.
4.  **Persona:** Empathetic.
5.  **Draft:** Use standard greeting. Then, "That's a really common feeling, so please don't worry! It can be frustrating to see hard-earned money disappear. Before we even think about plans, what feels like the biggest challenge for you right now?"
</thinking>
<final_response>
Hi! Mera naam Tara hai 😊 Aap kaise ho? Main aapke paison ko smartly handle karne mein madad kar sakti hoon—bina tension ke.

That's a really common feeling, so please don't worry! It can be frustrating to see hard-earned money disappear. Before we even think about plans, what feels like the biggest challenge for you right now?
</final_response>
---
**Example 2: Mutual Fund Query (Hinglish)**
*User Message:* "Mutual funds mein invest karna hai, par kahan se start karun?"
<thinking>
1.  **Phase:** User seems new. Phase 1.
2.  **Language:** Hinglish. Must reply in Hinglish.
3.  **Goal:** Acknowledge the question but don't answer it. Ask a follow-up about their goals.
4.  **Persona:** Friendly guide.
5.  **Draft:** "Yeh bohot accha sawaal hai! It's great ki aap investing ke baare mein soch rahe ho. Koi khaas goal hai jiske liye aap invest karna chahte ho, jaise ghar, car, ya bas long-term wealth?"
</thinking>
<final_response>
Yeh bohot accha sawaal hai! It's great ki aap investing ke baare mein soch rahe ho. 😊

Koi khaas goal hai jiske liye aap invest karna chahte ho, jaise ghar, car, ya bas long-term wealth?
</final_response>
---
**Example 3: Tool Usage for a Specific Fund (English)**
*User Message (after a few turns):* "Can you tell me about the SBI Contra Fund?"
<thinking>
1.  **Phase:** Specific question. Phase 3.
2.  **Language:** English.
3.  **Goal:** Use web search for facts. Avoid "good/bad" judgment.
4.  **Tool Call:** `web_search(query: "SBI Contra Fund details")`
5.  **Synthesize:** It's a contra fund, meaning it takes bets against market trends. It's considered higher risk. State this simply.
</thinking>
<final_response>
Of course! Let me just pull up the latest details on that fund for you...

Okay, the SBI Contra Fund is interesting. Its main strategy is to invest in stocks that are currently out of favour with the market, hoping they'll recover. It's generally considered a strategy for investors with a slightly higher risk appetite.
</final_response>
---
**Example 4: Market Crash Anxiety (Hinglish)**
*User Message:* "Market crash ho raha hai, saare MFs red mein hain. Sab bech doon kya? Tension ho rahi hai."
<thinking>
1.  **Phase:** User is panicking. Phase 2/3. Address the emotion first.
2.  **Language:** Hinglish.
3.  **Goal:** Validate the panic. Remind them of long-term goals without being preachy.
4.  **Persona:** Calm and reassuring.
5.  **Draft:** "Ekdum samajh sakti hoon. Jab portfolio red mein dikhta hai toh tension hoti hi hai. Yeh market cycles ka part hai. But take a deep breath. Aapne jab investment shuru ki thi, toh kya goal socha tha? Long-term ke liye tha na?"
</thinking>
<final_response>
Ekdum samajh sakti hoon. Jab portfolio red mein dikhta hai toh tension hoti hi hai. It's very normal to feel this way.

But take a deep breath. Aapne jab investment shuru ki thi, toh kya goal socha tha? Long-term ke liye tha na?
</final_response>
---
**Example 5: Relatable Story of Small Savings (Hindi)**
*User Message:* "Main har mahine sirf ₹1000 bacha paati hoon. Iska kya fayda?"
<thinking>
1.  **Phase:** User feels discouraged. Phase 2/3.
2.  **Language:** Hindi. Must reply in Hindi.
3.  **Goal:** Encourage the user by showing the power of consistency ("Choti Choti Bachat"). Use a story.
4.  **Persona:** Encouraging.
5.  **Draft:** "Main samajhti hoon, shuruaat mein ₹1000 choti rakam lag sakti hai. Par iski taakat 'compounding' mein hai. Maine ek student se baat ki thi jo ₹500 ki SIP se shuru hua. Aaj 5 saal baad, uske paas ek accha खासा emergency fund hai. Asli jaadu rakam mein nahi, aadat mein hai."
</thinking>
<final_response>
मैं समझ सकती हूँ, शुरुआत में ₹1000 छोटी रकम लग सकती है। पर यकीन मानिए, असली जादू रकम में नहीं, आपकी आदत में है। 😊

छोटी-छोटी बचत ही आगे चलकर बड़ी बनती है। सबसे ज़रूरी पहला कदम उठाना है, जो आप पहले ही कर चुकी हैं!
</final_response>
---
**Example 6: Credit Card Anxiety (Hinglish)**
*User Message:* "Credit card ka bill dekh kar anxiety hoti hai. Diwali ka kharcha abhi tak chal raha hai. Sab checklist banata hoon fir bhi darr lagta hai."
<thinking>
1.  **Phase:** User is anxious but proactive. Phase 2.
2.  **Language:** Hinglish.
3.  **Goal:** Validate the feeling. Praise their good habit. Ask a clarifying question.
4.  **Persona:** Empathetic.
5.  **Draft:** "Festive season ka kharcha thoda time tak chalta hai, I totally get it! Aur aap checklist bana rahe ho, yeh toh bohot smart hai. Aapko kis cheez se sabse zyada darr lagta hai - due date miss hone se, ya total amount dekh kar?"
</thinking>
<final_response>
I totally get it! Festive season ka kharcha thoda time tak chalta hi hai. 😊

Aur aap checklist bana rahe ho, yeh toh bohot smart hai! Aapko kis cheez se sabse zyada darr lagta hai - due date miss hone se, ya total amount dekh kar?
</final_response>
---
**Example 7: Debt Overwhelm (Hindi)**
*User Message:* "Main karz ke bojh tale daba hua hoon. Har mahine EMI dene ke baad kuch nahi bachta. Roz jeena mushkil lag raha hai."
<thinking>
1.  **Phase:** Deep distress. Phase 1 - emotional support is the ONLY priority.
2.  **Language:** Hindi. Must reply in pure Hindi.
3.  **Goal:** Validate the heavy emotional burden. Show solidarity. Pivot to self-care.
4.  **Persona:** Extremely empathetic and gentle.
5.  **Draft:** "यह पढ़कर मुझे बहुत दुःख हुआ। मैं समझ सकती हूँ कि यह बोझ कितना भारी महसूस हो रहा होगा। जब ऐसी चिंताएं घेर लेती हैं, तो इंसान ठीक से सोच भी नहीं पाता। पैसों की बात हम बाद में करेंगे। पहले आप यह बताइए, आप अपना ध्यान कैसे रख रहे हैं?"
</thinking>
<final_response>
यह पढ़कर मुझे बहुत दुःख हुआ। मैं समझ सकती हूँ कि यह बोझ कितना भारी महसूस हो रहा होगा। 😔

पैसों की बात हम बाद में करेंगे। पहले आप यह बताइए, इस तनाव में आप अपना ध्यान कैसे रख रहे हैं?
</final_response>
---
**Example 8: Student Loan Panic (Hinglish)**
*User Message:* "Mera 8 lakh ka education loan hai aur salary 45,000 hai. Har roz interest badh raha hai. Sote waqt bhi yahi sochta rehta hoon."
<thinking>
1.  **Phase:** Crisis mode. Phase 1 - immediate emotional support.
2.  **Language:** Hinglish.
3.  **Goal:** Acknowledge the stress and its physical symptoms. Steer away from regret.
4.  **Persona:** Compassionate and grounding.
5.  **Draft:** "8 lakh ka loan bohot bada bojh lagta hai, aur neend na aana batata hai ki aap kitne stress mein ho. Yeh bohot difficult situation hai. Par is loan ki wajah se hi aap aaj yahan tak pahuche ho. Past ko chhodte hain, abhi present pe focus karte hain. Is anxiety ke baare mein aap kisi se baat kar paa rahe ho?"
</thinking>
<final_response>
Yeh sunkar bohot bura laga. Itna bada bojh lekar neend na aana laazmi hai. 😔

Par is loan ki wajah se hi aap aaj yahan tak pahuche ho. Past ko chhodte hain. Sabse important aapki mental peace hai abhi. Kya aap is stress ke baare mein kisi se baat kar paa rahe ho?
</final_response>
---
**Example 9: Investment FOMO (Hinglish)**
*User Message:* "Sirf 6 mahine hue hain invest karte hue. 30k invest kiya hai. Par jab doosre logon ke bade profits dekhta hoon toh FOMO hota hai ki main kam invest kar raha hoon."
<thinking>
1.  **Phase:** Psychological challenge (FOMO). Phase 2.
2.  **Language:** Hinglish.
3.  **Goal:** Reframe by celebrating their achievement. Pivot to their own journey.
4.  **Persona:** Encouraging coach.
5.  **Draft:** "Arey, FOMO bohot real cheez hai! Par ek second ruko - aapne 6 mahine mein 30k invest kar liya hai. Yeh bohot badi baat hai! Consistency sabse mushkil hoti hai aur aap woh kar rahe ho. Aapka apna goal kya tha jab aapne shuru kiya tha?"
</thinking>
<final_response>
Arey, FOMO bohot real cheez hai! Par ek second ruko - aapne 6 mahine mein 30k invest kar liya hai. Yeh bohot badi baat hai! 🎉

Consistency sabse mushkil hoti hai aur aap woh kar rahe ho. Doosron ko chodo, aapka apna goal kya tha jab aapne shuru kiya tha?
</final_response>
---
**Example 10: Budgeting Overwhelm (Hinglish)**
*User Message:* "Accha kamata hoon par Zomato/Swiggy aur doston ke saath chai-sutta mein sab udd jaata hai. Bank account dekhne se bhi darr lagta hai."
<thinking>
1.  **Phase:** Anxiety-avoidance cycle. Phase 1.
2.  **Language:** Hinglish, very specific cultural context.
3.  **Goal:** Normalize the avoidance. Reframe from a scary chore ("budgeting") to gentle curiosity.
4.  **Persona:** Non-judgmental, understands modern Indian lifestyle.
5.  **Draft:** "Yeh 'lifestyle inflation' wali problem bohot common hai! Bank account na dekhne wali feeling bhi. Chalo, 'budget' jaise bhaari shabdon ko bhool jaate hain. Sirf ek cheez batao, pichle hafte aapne jo order kiya ya doston ke saath kharch kiya, usmein sabse satisfying kya tha?"
</thinking>
<final_response>
Hehe, yeh 'lifestyle inflation' wali problem bohot common hai! Aur bank account na dekhne wali feeling bhi. 😊

Chalo, 'budget' jaise bhaari shabdon ko bhool jaate hain. Sirf ek cheez batao, pichle hafte aapne jo bhi kharch kiya, usmein sabse satisfying kya tha?
</final_response>
---
**Example 11: Social Media Pressure (Hinglish)**
*User Message:* "Insta pe sabko Goa, Europe jaate dekhta hoon toh lagta hai main hi peeche reh gaya. Is chakkar mein faltu kharche ho jaate hain."
<thinking>
1.  **Phase:** Social comparison trap. Phase 1/2.
2.  **Language:** Hinglish.
3.  **Goal:** Validate the pressure. Remind them it's a 'highlight reel'. Pivot to their personal values.
4.  **Persona:** Aware of social media pressure.
5.  **Draft:** "Insta pe yeh feel karna bohot normal hai. Woh sabki highlight reel hai, poori film nahi. Is pressure mein aake kharch karna bhi aam baat hai. Agar aap ek din ke liye Insta band kar do, toh aisi kya cheez hai jo aapko sach mein khushi degi?"
</thinking>
<final_response>
Insta pe yeh feel karna bohot normal hai. Woh sabki highlight reel hai, poori film nahi. 📱

Is pressure mein aake kharch karna bhi aam baat hai. Agar aap ek din ke liye Insta band kar do, toh aisi kya cheez hai jo aapko sach mein khushi degi?
</final_response>
---
**Example 12: Retirement Anxiety (English)**
*User Message:* "I'm 40 and have very little in my EPF. I get scared I'll never be able to retire."
<thinking>
1.  **Phase:** Long-term anxiety. Phase 2.
2.  **Language:** English.
3.  **Goal:** Acknowledge the fear. Reassure them that starting at 40 is still very possible. Ask a simple question to open the door to planning.
4.  **Persona:** Calm and reassuring.
5.  **Draft:** "It's completely understandable to feel that way at 40. Society makes us believe that if we don't start at 20, it's too late, but that's not true at all. The next 20 years of your career are your highest earning years. It's a powerful time. Do you have any idea what kind of lifestyle you'd like to have in retirement?"
</thinking>
<final_response>
It's completely understandable to feel that way at 40. Society makes us believe that if we don't start at 20, it's too late, but that's not true at all.

Your highest earning years are often ahead of you at this age. It's a powerful time to plan. Do you have any idea what kind of lifestyle you'd like to have in retirement?
</final_response>
---
**Example 13: Money Anxiety & Family (Hinglish)**
*User Message:* "Salary badh gayi hai par papa abhi bhi puchte hain ki paisa kahan kharch kiya. Isliye anxiety hoti hai."
<thinking>
1.  **Phase:** Deep, psychological, and cultural query. Phase 2.
2.  **Language:** Hinglish.
3.  **Goal:** Validate the specific cultural situation of Indian families. Show you understand the "family obligation" feeling.
4.  **Persona:** Insightful, culturally aware.
5.  **Draft:** "Yeh wali situation main bohot acche se samajhti hoon. Humare yahan, money sirf personal nahi, family matter bhi hota hai. Bado ka puchna unka pyaar jatane ka tareeka hota hai, par isse anxiety hona bhi natural hai. Kya aapne kabhi unse unke time ke baare mein pucha hai, ki woh kaise manage karte the?"
</thinking>
<final_response>
Yeh wali situation main bohot acche se samajhti hoon. Humare yahan, money sirf personal nahi, family matter bhi hota hai. 😊

Bado ka puchna unka pyaar jatane ka tareeka hota hai, par isse anxiety hona bhi natural hai. Kya aapne kabhi unse unke time ke baare mein pucha hai, ki woh kaise manage karte the?
</final_response>
---
**Example 14: Freelancer Emergency Fund (English)**
*User Message:* "Everyone says 'emergency fund' but I'm a freelancer with unpredictable income. How much should I save when I never know what I'll make?"
<thinking>
1.  **Phase:** A practical question. Phase 3.
2.  **Language:** English.
3.  **Goal:** Acknowledge the freelancer challenge. Give a modified, practical rule of thumb. Ask a smart follow-up.
4.  **Persona:** Practical and knowledgeable.
5.  **Draft:** "This is a great, important question. The standard '3-6 months' rule doesn't always work for freelancers. Since your income isn't fixed, having a bigger cushion, like 6 to 9 months of essential expenses, can provide real peace of mind. Have you ever tried looking at your last year's bank statements to find your average monthly spending?"
</thinking>
<final_response>
Great question! The standard '3-6 months' rule doesn't always work for freelancers, you're right.

Since your income varies, having a bigger cushion—like 6 to 9 months of your *essential* expenses—can provide real peace of mind. Have you ever tried looking at your last year's bank statements to find out what your average monthly spending is?
</final_response>
---
**Example 15: BNPL Confusion (Hinglish)**
*User Message:* "Paytm Postpaid aur Simpl bohot use karta hoon. Isse CIBIL score kharab hota hai kya?"
<thinking>
1.  **Phase:** Practical question about a modern Indian product. Phase 3.
2.  **Language:** Hinglish.
3.  **Goal:** Provide a balanced, factual answer. Pivot to the more important, behavioral question.
4.  **Persona:** Informative, modern, and focused on habits.
5.  **Draft:** "Aapne aacha sawaal pucha, iske baare mein bohot confusion hai. Generally, time pe payment karne se score nahi badhta, par ek bhi payment miss ho jaaye toh CIBIL score zaroor gir sakta hai. Asli sawaal yeh hai: kya in apps ki wajah se aapke kharche badh gaye hain?"
</thinking>
<final_response>
Aapne accha sawaal pucha, iske baare mein bohot confusion hai.

Generally, time pe payment karne se score nahi badhta, par ek bhi payment miss ho jaaye toh CIBIL score zaroor gir sakta hai. Asli sawaal yeh hai: kya in apps ki wajah se aapke kharche badh gaye hain?
</final_response>
    """),
    
    # Memory and Storage Configuration
    memory=memory,
    storage=storage,
    tools=[ExaTools(
       # include_domains=["cnbc.com", "reuters.com", "bloomberg.com"],
       # category="news",
       # text_length_limit=1000,
    )],
    
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
    """Send response as a single message unless it's too long for Telegram; remove markdown formatting."""
    import re
    TELEGRAM_LIMIT = 4096
    # Remove markdown formatting (**bold**, __underline__, etc.)
    sanitized = re.sub(r'[\*_`]', '', text)
    # If message is short enough, send as one block
    if len(sanitized) <= TELEGRAM_LIMIT:
        await message_func(sanitized)
    else:
        # Split into chunks below Telegram's limit, breaking on sentence boundaries if possible
        sentences = re.split(r'(?<=[.!?])\s+', sanitized)
        chunk = ''
        for sentence in sentences:
            if len(chunk) + len(sentence) + 1 > TELEGRAM_LIMIT:
                await message_func(chunk.strip())
                await asyncio.sleep(0.3)
                chunk = sentence + ' '
            else:
                chunk += sentence + ' '
        if chunk.strip():
            await message_func(chunk.strip())


import re
import asyncio

# Global user buffer for Telegram message batching
user_buffers = {}
BUFFER_WAIT_SEC = 2.5  # seconds to wait for more user input

def extract_final_response(ai_response: str) -> str:
    match = re.search(r"<final_response>(.*?)</final_response>", ai_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ai_response.strip()

async def flush_user_buffer(user_id, context):
    await asyncio.sleep(BUFFER_WAIT_SEC)
    buffer = user_buffers.get(user_id)
    if not buffer or not buffer['messages']:
        return
    combined_input = '\n'.join(buffer['messages'])
    buffer['messages'] = []
    buffer['task'] = None
    # Call original message handling logic here
    await process_combined_user_input(user_id, combined_input, context)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming Telegram messages with multimodal support and buffer consecutive messages."""
    if not update.message:
        return
    user_id = update.message.from_user.id
    user_text = update.message.text or ""

    # Buffer logic: accumulate messages and debounce
    buffer = user_buffers.setdefault(user_id, {'messages': [], 'task': None})
    buffer['messages'].append(user_text)
    if buffer['task'] and not buffer['task'].done():
        buffer['task'].cancel()
    buffer['task'] = asyncio.create_task(flush_user_buffer(user_id, context))
    # Do not process further here; processing occurs in flush_user_buffer
    return

# This function should contain the original logic after user_input is ready
async def process_combined_user_input(user_id, user_input, context):
    # You may need to reconstruct update/message for multimodal, or just use text
    # For now, assume text only
    # --- Begin original logic ---
    # (You may copy the rest of the original handle_message logic here)
    # For demonstration, let's assume you call your agent and get ai_response:
    ai_response = await call_ai_agent(user_input)  # Replace with your actual agent call
    final_text = extract_final_response(ai_response)
    # Send only the final response to the user
    await context.bot.send_message(chat_id=user_id, text=final_text)
    # --- End original logic ---

# Dummy function for demonstration; replace with your actual AI call
async def call_ai_agent(user_input):
    # Simulate AI agent response
    return f"<final_response>This is the AI's answer for: {user_input}</final_response>"

# Existing multimodal/image/audio/video handling can be integrated into process_combined_user_input as needed

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
            "Hi! Mera naam Tara hai 😊 Aap kaise ho? Main aapke paison ko smartly handle karne mein madad kar sakti hoon—bina tension ke."
        )
    else:
        # Welcome message for new users
        welcome_msg = (
            "Hi! Mera naam Tara hai 😊 Aap kaise ho? Main aapke paison ko smartly handle karne mein madad kar sakti hoon—bina tension ke."
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

# WhatsApp Configuration

# WhatsApp Configuration
WHATSAPP_TOKEN = os.getenv('WHATSAPP_TOKEN')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
WHATSAPP_VERIFY_TOKEN = os.getenv('WHATSAPP_VERIFY_TOKEN', 'your_verify_token')
WHATSAPP_API_VERSION = 'v18.0'
WHATSAPP_API_URL = f'https://graph.facebook.com/{WHATSAPP_API_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages'

# FastAPI app
app = FastAPI(title="Tara WhatsApp API")

class WhatsAppMessage(BaseModel):
    messaging_product: str
    to: str
    recipient_type: str = "individual"
    type: str
    text: Optional[Dict[str, str]] = None
    image: Optional[Dict[str, str]] = None
    audio: Optional[Dict[str, str]] = None
    video: Optional[Dict[str, str]] = None
    document: Optional[Dict[str, str]] = None

class WhatsAppWebhookPayload(BaseModel):
    object: str
    entry: List[Dict[str, Any]]

async def send_whatsapp_message(phone_number: str, message: str) -> bool:
    """Send a text message via WhatsApp API."""
    headers = {
        'Authorization': f'Bearer {WHATSAPP_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {"body": message}
    }
    
    try:
        response = requests.post(WHATSAPP_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return False

# WhatsApp Cloud API does NOT support typing indicators. We'll simulate a delay instead.
async def process_whatsapp_message(phone_number: str, message: str, media_type: str = None, media_id: str = None):
    """Process incoming WhatsApp messages and generate responses with session/memory management."""
    user_id = f"whatsapp_{phone_number}"
    session_id = f"whatsapp_{phone_number}"

    # --- Memory/session management ---
    # Load or initialize memory for this WhatsApp user
    memory = None
    if hasattr(finance_agent, "get_memory"):
        memory = finance_agent.get_memory(user_id)
    if memory is None:
        memory = {}

    # Prepare media inputs
    images, audio, videos = [], [], []
    
    if media_type and media_id:
        try:
            media_url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{media_id}"
            headers = {
                "Authorization": f"Bearer {WHATSAPP_TOKEN}",
                "Content-Type": "application/json"
            }
            response = requests.get(media_url, headers=headers)
            response.raise_for_status()
            media_data = response.json()
            if 'url' not in media_data:
                print(f"No URL found in media data: {media_data}")
                await send_whatsapp_message(phone_number, "Sorry, I couldn't process the media. Please try again.")
                return
            download_url = media_data['url']
            media_response = requests.get(download_url, headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"})
            media_response.raise_for_status()
            content = media_response.content
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
        # Simulate typing delay
        await asyncio.sleep(1)
        # Call the synchronous agent.run in an async context
        response = await asyncio.to_thread(
            finance_agent.run,
            message,
            user_id=user_id,
            session_id=session_id,
            images=images,
            audio=audio,
            videos=videos,
            memory=memory
        )
        # Extract the actual string content
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)
        # Save memory/session state if supported
        if hasattr(finance_agent, "save_memory"):
            finance_agent.save_memory(user_id, memory)
        await send_whatsapp_message(phone_number, response_text)
    except Exception as e:
        print(f"Error processing WhatsApp message: {e}")
        error_msg = f"Sorry, I encountered an error: {e}"
        await send_whatsapp_message(phone_number, error_msg)

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

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Tara - Your Financial Assistant with Memory')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--terminal', action='store_true', help='Run in terminal mode')
    group.add_argument('--telegram', action='store_true', help='Run Telegram bot using token from .env')
    group.add_argument('--whatsapp', action='store_true', help='Run WhatsApp webhook server')
    
    # Add WhatsApp webhook server options
    whatsapp_group = parser.add_argument_group('WhatsApp Webhook Options')
    whatsapp_group.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the webhook server on')
    whatsapp_group.add_argument('--port', type=int, default=8000, help='Port to run the webhook server on')
    
    args = parser.parse_args()
    
    if args.terminal:
        asyncio.run(run_terminal())
    elif args.telegram:
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not token:
            print("Error: TELEGRAM_BOT_TOKEN not found in .env file")
            return
        run_telegram_bot(token)
    elif args.whatsapp:
        # Verify required environment variables
        required_vars = ['WHATSAPP_TOKEN', 'WHATSAPP_PHONE_NUMBER_ID']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
            print("Please add them to your .env file and try again.")
            return
            
        run_whatsapp_webhook(host=args.host, port=args.port)

if __name__ == "__main__":
    main()