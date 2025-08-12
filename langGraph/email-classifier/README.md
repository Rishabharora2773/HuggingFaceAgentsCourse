# Email Classifier with LangGraph

An intelligent email classification system built with LangGraph that uses a local Qwen2.5 model via Ollama to automatically classify emails as spam or legitimate, and draft appropriate responses.

## Features

- **Email Classification**: Automatically determines if emails are spam or legitimate
- **Spam Detection**: Identifies spam emails and provides reasoning
- **Response Drafting**: Generates professional draft responses for legitimate emails
- **Local AI Model**: Uses Qwen2.5 running locally via Ollama
- **Workflow Visualization**: Generates a visual graph of the email processing workflow

## Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally
- Qwen2.5 model pulled in Ollama

## Installation

### 1. Create and Activate Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install langchain-community langgraph langchain-core
```

### 3. Set Up Ollama

```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai for installation instructions

# Start Ollama server
ollama serve

# Pull the Qwen2.5 model
ollama pull qwen2.5:latest
```

## Usage

### 1. Start Ollama Server

Make sure Ollama is running on localhost:11434:

```bash
ollama serve
```

### 2. Run the Email Classifier

```bash
python graph.py
```

The script will:
- Process sample legitimate and spam emails
- Classify each email using the Qwen2.5 model
- Generate draft responses for legitimate emails
- Create a visual graph representation (`graph.png`)

## How It Works

The email classifier uses a LangGraph workflow with the following nodes:

1. **read_email**: Logs and preprocesses incoming emails
2. **classify_email**: Uses Qwen2.5 to determine if email is spam or legitimate
3. **handle_spam**: Processes spam emails (moves to spam folder)
4. **draft_response**: Generates professional draft responses for legitimate emails
5. **notify_mr_hugg**: Presents the results and draft response

### Email Classification Logic

- **Spam Detection**: Looks for "SPAM" keyword in model response
- **Reason Extraction**: Extracts spam reasoning after "reason:" or "Reason:"
- **Category Classification**: Categorizes legitimate emails (inquiry, complaint, thank you, etc.)

## Sample Output

```
Alfred is processing an email from john.smith@example.com with subject: Question about your services
spam or not: False
spam reason: None

Processing legitimate email...
Alfred is processing an email from winner@lottery-intl.com with subject: YOU HAVE WON $5,000,000!!!
spam or not: True
spam reason: This email contains suspicious lottery claims and requests for personal financial information

Processing spam email...
```

## Configuration

The model configuration can be modified in `graph.py`:

```python
model = Ollama(
    model="qwen2.5:latest",  # Change model as needed
    base_url="http://localhost:11434",  # Ollama server URL
    temperature=0  # Adjust creativity (0 = deterministic, higher = more creative)
)
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**: Ensure `ollama serve` is running on localhost:11434
2. **Model Not Found**: Run `ollama pull qwen2.5:latest` to download the model
3. **Import Errors**: Make sure virtual environment is activated and dependencies are installed
4. **Permission Errors**: Ensure you have write permissions in the current directory
