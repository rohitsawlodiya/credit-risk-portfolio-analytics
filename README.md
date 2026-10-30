## Quick Start

Requires Python 3.12.

1. Install dependencies:
   `pip install -r requirements.txt`

2. Build the processed data and model:
   `python -m src.pipeline`

3. Start the dashboard:
   `streamlit run app/app.py`

## LLM Configuration

Set `OPENAI_API_KEY` privately in your `.env` file
or hosting platform's secrets store.

The application uses `gpt-4.1-mini` by default.
Without an API key, the dashboard remains available,
but the LLM assistant cannot answer questions.