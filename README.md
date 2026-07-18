# AI-Research-Paper-Intelligent-System-using-NLP-Transformers-Semantic-Search-and-Agentic-AI


An AI-powered system to search, understand, and interact with research papers using **Semantic Search, Summarization, Keyword Extraction, NER, and a LangChain Agent**.

---

## Overview
Reading 100s of research papers is slow. This project makes it faster.

The system uses **Sentence Transformers + FAISS** for semantic search, **BART** for summarization, **KeyBERT** for keywords, and a **LangChain + Groq LLM agent** to act as a research assistant. 

Instead of keyword matching, it understands the meaning of your query and retrieves relevant papers from the arXiv ML dataset. It can also summarize, compare papers, extract entities, and remember your previous searches.

---

## Key Features
- **Semantic Search**: Find papers based on meaning, not just keywords using FAISS + `all-MiniLM-L6-v2`
- **AI Summarization**: Auto-generate concise abstract summaries using `facebook/bart-large-cnn`
- **Keyword Extraction**: Extract top keywords and keyphrases with KeyBERT
- **Named Entity Recognition**: Identify models, datasets, institutions, etc using HuggingFace NER pipeline
- **Paper Comparison**: Compare 2 papers on objective, method, pros/cons using Groq LLM
- **Agentic Workflow**: LangChain agent decides which tool to use based on your prompt
- **Conversational Memory**: Follow-up questions work because the agent remembers your last search

---

## Dataset
**Source**: `CShorten/ML-ArXiv-Papers` from Hugging Face  
**Used**: `title` + `abstract` columns from first 15,000 papers  
**Preprocessing**: Combined into `paper_text` column. Cleaned newlines and extra spaces.

---

## Tech Stack
**Core**: Python, Pandas, NumPy  
**Embeddings & Search**: Sentence-Transformers, Scikit-learn, FAISS  
**NLP Models**: Hugging Face Transformers - BART, NER Pipeline, KeyBERT  
**Agent Framework**: LangChain, LangGraph  
**LLM**: Groq `llama-3.1-8b-instant`

---

## Models Used

| Model | Role |
| --- | --- |
| `all-MiniLM-L6-v2` | Convert paper text to semantic embeddings |
| `facebook/bart-large-cnn` | Summarize abstracts |
| HuggingFace NER Pipeline | Extract entities like PyTorch, ResNet, Stanford |
| `llama-3.1-8b-instant` via Groq | Agent reasoning + Paper comparison |

---

## How It Works
1.  **Load Data**: Load 15k ML papers from HuggingFace dataset
2.  **Create Embeddings**: Generate embeddings with Sentence Transformers and save to `paper_embeddings.npy`
3.  **Build Index**: Store normalized embeddings in FAISS `IndexFlatIP` as `paper_faiss.index`
4.  **Query**: User query → embedding → FAISS similarity search → top-k papers
5.  **Post-Process**: Use BART to summarize, KeyBERT to extract keywords
6.  **Agent**: LangChain tools wrap each function. Agent + Groq LLM picks the right tool
7.  **Memory**: `MemorySaver` + `paper_memory` stores last search for follow-ups

---

## Agent Tools
1.  **search_and_summarize**: Search papers and return summaries
2.  **extract_keywords**: Get top keywords/phrases from any text
3.  **compare_papers**: Semantic search 2 topics → summarize both → LLM comparison
4.  **get_last_search**: Fetch papers from the previous query

---

## API Key Setup
The Agentic AI section requires a Groq API key.

To use it in Google Colab:

1. Open the Secrets section.
2. Add your Groq API key.
3. Save it with the name GROQ_API_KEY.
4. Allow notebook access to the secret.

The API key is then securely accessed from Colab Secrets instead of being directly written in the notebook.

---

## Conclusion
This project presents an Agentic AI-powered Research Paper Intelligence System that combines NLP, Transformers, Semantic Search, and LLMs. 

By using Sentence Transformers with FAISS for retrieval, BART for summarization, KeyBERT for keywords, and a LangChain Agent with Groq LLM for decision-making, the system makes exploring research papers faster and more interactive.

With conversational memory, users can ask follow-up questions without repeating context. Overall, this system shows how modern AI techniques can work together to help researchers find and understand papers efficiently.
