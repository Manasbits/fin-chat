# Tara - Financial Assistant with Memory

Tara is a smart AI-based financial assistant that can help you with various financial tasks. It supports multiple platforms including Terminal, Telegram, and WhatsApp.

## Features

- **Multimodal Input**: Supports text, images, audio, and document inputs
- **Memory**: Remembers previous conversations and context
- **Multi-platform**: Available on Terminal, Telegram, and WhatsApp
- **Docker Support**: Easy deployment with Docker

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- API keys for the services you want to use (Telegram, WhatsApp)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd skite-basic
   ```

2. Create a `.env` file in the root directory with the following variables:
   ```env
   # Telegram
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   
   # WhatsApp
   WHATSAPP_TOKEN=your_whatsapp_business_token
   WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
   WHATSAPP_VERIFY_TOKEN=your_webhook_verify_token
   
   # OpenAI (if using)
   OPENAI_API_KEY=your_openai_api_key
   
   # Google AI (if using)
   GOOGLE_API_KEY=your_google_api_key
   ```

## Running with Docker (Recommended)

1. Build and start the services:
   ```bash
   # Start both Telegram and WhatsApp
   docker-compose up --build
   
   # Or start specific services
   docker-compose up --build telegram-bot
   docker-compose up --build whatsapp-webhook
   ```

2. The services will be available at:
   - Telegram bot: Running and ready to receive messages
   - WhatsApp webhook: http://localhost:8000/webhook

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the desired service:
   ```bash
   # Terminal mode
   python agent.py --terminal
   
   # Telegram bot
   python agent.py --telegram
   
   # WhatsApp webhook
   python agent.py --whatsapp --host 0.0.0.0 --port 8000
   
   # Run both Telegram and WhatsApp
   python agent.py --telegram --whatsapp --host 0.0.0.0 --port 8000
   ```

## WhatsApp Webhook Setup

1. Deploy the webhook to a public URL (using ngrok, for example):
   ```bash
   ngrok http 8000
   ```

2. Configure the webhook in your WhatsApp Business API settings:
   - URL: `https://your-ngrok-url.ngrok.io/webhook`
   - Verify Token: The one you set in `.env` as `WHATSAPP_VERIFY_TOKEN`

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token | For Telegram |
| `WHATSAPP_TOKEN` | Your WhatsApp Business API token | For WhatsApp |
| `WHATSAPP_PHONE_NUMBER_ID` | Your WhatsApp Business phone number ID | For WhatsApp |
| `WHATSAPP_VERIFY_TOKEN` | Webhook verification token | For WhatsApp |
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI integration |
| `GOOGLE_API_KEY` | Google AI API key | For Google AI integration |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
