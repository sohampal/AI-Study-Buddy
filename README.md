# 🤖 AI Study Buddy

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?logo=streamlit)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-purple)
![ChromaDB](https://img.shields.io/badge/Database-ChromaDB-green)
![LLM](https://img.shields.io/badge/LLM-LLaMA3-orange)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

An intelligent AI-powered chatbot designed to help students learn physics through interactive conversations, problem-solving, and concept explanations.


## 📌 Overview

AI Study Buddy is a smart educational assistant that combines **Large Language Models (LLMs)** with **Retrieval-Augmented Generation (RAG)** and **agent-based workflows** to provide accurate and context-aware responses.


## 🚀 Features

* 🔀 Smart Query Routing
* 📚 RAG-Based Knowledge System
* 🧠 Context Awareness
* 🔍 Tool Integration
* ✅ Self-Evaluation System

---

## 🏗️ System Architecture

```
User Input → Router → (Retriever / Tool / Memory)
                  ↓
                LLM
                  ↓
            Evaluation
                  ↓
            Final Response
```

---

## 🛠️ Tech Stack

| Component       | Technology            |
| --------------- | --------------------- |
| Frontend        | Streamlit             |
| Backend         | Python                |
| LLM             | Groq (LLaMA 3.1)      |
| Embeddings      | Sentence Transformers |
| Vector Database | ChromaDB              |
| Agent Framework | LangGraph             |

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/AI-Study-Buddy.git
cd AI-Study-Buddy
pip install -r requirements.txt
```

Create `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

Run:

```bash
streamlit run app.py
```

---

## 💡 Usage

* Ask physics questions
* Solve numerical problems
* Explore concepts interactively

---

## 📊 Advantages

* Intelligent multi-route system
* Context-aware responses
* Interactive learning experience

---

## ⚠️ Limitations

* Limited knowledge base
* API dependency
* Not real-time optimized

---

## 🔮 Future Scope

* Multi-subject support
* Better UI/UX
* Voice interaction
* Full deployment

---

## 📄 Project Report

`AI_Study_Buddy_Report.pdf`

---

## 👨‍💻 Author

**Your Name**
GitHub: https://github.com/your-username
LinkedIn: https://linkedin.com/in/your-profile

---

## ⭐ Contribute

Feel free to fork and submit pull requests!

---

## 📜 License

This project is for educational purposes.
