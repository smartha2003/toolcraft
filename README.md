---
title: ToolCraft Agent
emoji: ⚡
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
tags:
  - smolagents
  - agent
  - code-agent
  - ai
  - python
  - image-generation
  - web-search
---

# 🚀 ToolCraft Agent

> An intelligent AI agent that writes and executes Python code to solve complex tasks using web search, image generation, and custom tools.

[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/smartha2003/toolcraft-agent)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.23+-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ Features

- **🧠 Code Generation**: AI agent writes and executes Python code dynamically to solve tasks
- **🌐 Web Search**: Integrated DuckDuckGo search for real-time information retrieval
- **📄 Web Scraping**: Visit and extract content from webpages with markdown conversion
- **🎨 Image Generation**: Generate images from text prompts using Hugging Face's image models
- **💬 Conversational Memory**: Maintains context and variables across multiple messages
- **🔄 Multi-step Reasoning**: Breaks down complex tasks into sequential steps
- **📊 Execution Logs**: Transparent view of code execution and tool outputs
- **🛡️ Error Handling**: Graceful error recovery and informative error messages

## 🎯 Live Demo

Try it out: **[https://huggingface.co/spaces/smartha2003/toolcraft-agent](https://huggingface.co/spaces/smartha2003/toolcraft-agent)**

## 🏗️ Architecture

### How It Works

```
User Query
    ↓
LLM (Qwen2.5-Coder) generates reasoning
    ↓
LLM generates Python code with tool calls
    ↓
Code executes → Tools called → Results observed
    ↓
LLM processes results → Next step or final_answer()
    ↓
Result displayed in chat interface
```

### Key Components

- **CodeAgent**: Core agent that orchestrates code generation and execution
- **Tools**: Modular Python functions (web_search, visit_webpage, image_generator)
- **Gradio UI**: Interactive chat interface with streaming responses
- **Memory System**: Persistent state across conversation turns

## 🛠️ Technology Stack

- **Framework**: [smolagents](https://github.com/huggingface/smolagents) by Hugging Face
- **LLM**: Qwen/Qwen2.5-Coder-32B-Instruct
- **UI**: Gradio
- **Tools**: DuckDuckGo Search, Requests, PIL/Pillow
- **Deployment**: Hugging Face Spaces

## 📦 Installation

### Prerequisites

- Python 3.10+
- Hugging Face account with API token

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/smartha2003/toolcraft.git
   cd toolcraft
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file:
   ```bash
   HF_TOKEN=your_huggingface_token_here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

The app will launch at `http://localhost:7860`

## 🎮 Usage Examples

### Image Generation
```
User: "Generate an image of a black cat with green eyes"
Agent: [Generates and displays image]
```

### Web Research
```
User: "What is the population of Tokyo?"
Agent: [Searches web, extracts information, provides answer]
```

### Multi-step Tasks
```
User: "Find the current time in San Francisco and convert it to UTC"
Agent: [Searches for time → Converts timezone → Provides answer]
```

## 📁 Project Structure

```
toolcraft-agent/
├── app.py                 # Main entry point, agent initialization
├── Gradio_UI.py          # Chat interface and message streaming
├── prompts.yaml          # System prompts and LLM instructions
├── agent.json            # Agent configuration
├── requirements.txt     # Python dependencies
├── tools/
│   ├── final_answer.py  # Final answer tool (handles PIL Images)
│   ├── web_search.py    # DuckDuckGo search implementation
│   └── visit_webpage.py # Webpage fetching and parsing
└── README.md            # This file
```

## 🔧 Key Features Explained

### 1. Dynamic Code Execution
The agent writes Python code on-the-fly based on the task, executing it in a sandboxed environment with access to custom tools.

### 2. Persistent Memory
Variables and imports persist across steps, allowing the agent to build on previous work:
```python
# Step 1
import pandas as pd
data = [1, 2, 3]

# Step 2 (can use previous variables!)
df = pd.DataFrame(data)  # ✅ Still available
```

### 3. Image Generation Pipeline
- Generates images using Hugging Face's text-to-image models
- Converts PIL Images to file format automatically
- Displays images inline in the chat interface

### 4. Error Recovery
- Graceful handling of failed web searches
- Automatic retry with different strategies
- Informative error messages for debugging

## 🚀 Deployment

### Hugging Face Spaces

1. Push code to Hugging Face Space
2. Add `HF_TOKEN` as a Secret in Space settings
3. Space auto-rebuilds on push

### Local Deployment

```bash
python app.py
```

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | Hugging Face API token | Yes |
| `HUGGINGFACE_HUB_TOKEN` | Alternative token name | Optional |

## 📝 Configuration

Key settings in `app.py`:
- `max_steps`: Maximum reasoning steps (default: 10)
- `model_id`: LLM model identifier
- `temperature`: Model temperature (default: 0.5)

## 🐛 Troubleshooting

### API Rate Limits
If you hit rate limits, consider:
- Using a smaller model
- Adding credits to your Hugging Face account
- Using a different model endpoint

### Image Not Displaying
- Ensure `HF_TOKEN` is set correctly
- Check that image generation tool is loaded
- Verify PIL/Pillow is installed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Shubhada Martha**
- GitHub: [@smartha2003](https://github.com/smartha2003)
- Hugging Face: [@smartha2003](https://huggingface.co/smartha2003)

## 🙏 Acknowledgments

- Built with [smolagents](https://github.com/huggingface/smolagents) framework
- Powered by [Hugging Face](https://huggingface.co/)
- UI built with [Gradio](https://gradio.app/)

---

⭐ If you find this project interesting, please give it a star!
