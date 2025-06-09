# IngredientCheck

AI-powered ingredient analysis platform that provides personalized nutrition insights based on your health profile. Upload photos of ingredient lists and get instant health recommendations tailored to your allergies and medical conditions.

## Features

- **AI-Powered OCR**: Extract ingredients from photos using PaddleOCR
- **Personalized Analysis**: Get health insights based on your allergies and medical history
- **Smart Recommendations**: Receive safer product alternatives
- **Analysis History**: Track your ingredient analysis journey
- **Medical Profile**: Secure storage of health information

## Quick Start

### Clone and Setup

```bash
git clone https://github.com/durgeshmehar/Ingredient-Check

cd Ingredient-Check
```

### Build Docker Image

```bash
docker build -t ingredient-app .
```

### Run Application with Persistent Database

```bash
docker run -p 8000:8000 \
  -v $(pwd)/db.sqlite3:/app/db.sqlite3 \
  -e DEBUG=False \
  -e GROQ_API_KEY=your_groq_api_key \
  -e LANGCHAIN_API_KEY=your_langchain_api_key \
  ingredient-app
```

**Note**: 
- Replace `your_groq_api_key` and `your_langchain_api_key` with your actual API keys

### Access Application

Open your browser and navigate to `http://0.0.0.0:8000`

## Environment Variables

- `GROQ_API_KEY`: Required for AI analysis
- `LANGCHAIN_API_KEY`: Required for LangChain tracing  
- `DEBUG`: Set to `False` for production

## Database Persistence

The application uses a Docker volume to persist the SQLite database between container restarts. Your user registrations and analysis history will be maintained across container lifecycles.

## Tech Stack

- **Backend**: Django 5.1
- **AI/ML**: LangChain, Groq LLM, PaddleOCR
- **Frontend**: Tailwind CSS, Font Awesome
- **Database**: SQLite (with Docker volume persistence)
- **Deployment**: Docker

## Developed By

Durgesh Mehar and team members