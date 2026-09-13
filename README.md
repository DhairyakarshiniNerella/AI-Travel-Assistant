# ✈️ AI Travel Assistant

An AI-powered travel assistant that combines **RAG (Retrieval-Augmented Generation)** and **Agentic AI** to provide travel-related information.

The application can answer questions from a travel knowledge base and use external tools for information such as **weather and places**.

## 🚀 Features

* 🤖 AI-powered travel assistant
* 🔎 RAG-based document retrieval
* 🧠 LangGraph agent workflow
* 📚 Travel knowledge base
* 🖼️ Vision-LLM image understanding (identifies landmarks/subjects in photos)
* 🔢 HuggingFace embeddings
* 🗄️ ChromaDB vector database
* 🌦️ Weather tool
* 📍 Places tool
* 💬 Streamlit web interface
* ⚡ Groq LLM integration

## 🏗️ Architecture

```text
User
 |
 v
Streamlit UI
 |
 v
LangGraph Agent
 |
 +------------+------------+
 |            |            |
 v            v            v
RAG Tool   Places Tool  Weather Tool
 |            |            |
 v            v            v
ChromaDB   Places API   Weather API
 |
 v
Groq LLM
 |
 v
Final Response
```

## 🔄 RAG Pipeline

```text
Documents
    |
    v
Document Loading
    |
    v
Text Extraction (PDF/DOCX/TXT/XLSX)
or Vision-LLM Description (images)
    |
    v
Chunking
    |
    v
HuggingFace Embeddings
    |
    v
ChromaDB
    |
    v
Similarity Search
    |
    v
Relevant Context
    |
    v
Groq LLM
    |
    v
Answer
```

## 🛠️ Technologies

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Application development |
| Streamlit    | User interface          |
| LangChain    | RAG and LLM integration |
| LangGraph    | Agent workflow          |
| Groq         | LLM inference           |
| HuggingFace  | Text embeddings         |
| ChromaDB     | Vector database         |
| Groq Vision LLM | Image understanding (landmark/subject identification) |
| Git & GitHub | Version control         |

## 📂 Project Structure

```text
AI-Travel-Assistant/
│
├── app.py
│
├── agents/
│   ├── travel_agent.py       (main LangGraph agent used by app.py)
│   ├── weather_agent.py
│   └── places_agent.py
│
├── tools/
│   ├── weather_tool.py
│   ├── places_tool.py
│   └── rag_tool.py
│
├── services/
│   ├── geocoding_service.py
│   ├── weather_service.py
│   └── places_service.py
│
├── rag/
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── image_vision.py
│   ├── retriever.py
│   └── vector_store.py
│
├── data/
│   └── documents/
│       ├── travel_guide_india.txt
│       ├── travel_guide_india.pdf
│       ├── travel_guide_india.docx
│       ├── travel_guide_india.xlsx
│       └── travel_guide_india.png
│
├── requirements.txt
├── packages.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🧠 Embedding Model

The project uses:

```text
all-MiniLM-L6-v2
```

It generates **384-dimensional embeddings** that are stored in ChromaDB.

## 🗄️ Vector Database

**ChromaDB** is used to store:

* Document chunks
* Embeddings
* Metadata

Collection name:

```text
travel_knowledge
```

## 🖼️ Image Understanding

Uploaded travel photos are processed by a vision-capable Groq LLM, which identifies the landmark or subject and generates a short description.

```text
Image
 |
 v
Vision LLM (Groq)
 |
 v
Generated Description
 |
 v
Chunking
 |
 v
Embeddings
 |
 v
ChromaDB
```

This allows the content of uploaded photos to become searchable through the RAG pipeline.

Note: this is a vision-language model describing what it sees, not OCR — it's not designed to transcribe dense or small text in an image character-for-character.

## 🤖 Agentic AI

The LangGraph agent decides which tool should be used based on the user's question.

For example:

```text
"What is the weather in Hyderabad?"
        |
        v
   Weather Tool
        |
        v
   Weather API
```

or:

```text
"Tell me about Charminar"
        |
        v
      RAG Tool
        |
        v
     ChromaDB
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd AI-Travel-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API Key

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GEOAPIFY_API_KEY=your_geoapify_api_key
```

Do not commit `.env` to GitHub.

### 6. Run the application

```bash
python -m streamlit run app.py
```




## 💡 Example Questions

```text
What are the famous places in Hyderabad?

Tell me about Charminar.

Find restaurants near Charminar.

What is the weather in Hyderabad?

Suggest places to visit in Hyderabad.
```

## 🎯 Project Objective

The main objective of this project is to gain practical experience with:

* RAG
* Vector databases
* Embeddings
* Semantic search
* Vision-language models for image understanding
* LangChain
* LangGraph
* Agentic AI
* Tool calling
* LLM integration
* Streamlit

## 📌 Project Status

**Status: 🚧 In Development**

Core functionality has been implemented, including RAG, ChromaDB, HuggingFace embeddings, vision-LLM image understanding, LangGraph agent, weather tool, places tool, Groq LLM, and Streamlit UI.

## 👩‍💻 Author

**AI Travel Assistant**

Built as a hands-on **Generative AI project** to learn and demonstrate RAG and Agentic AI concepts.
