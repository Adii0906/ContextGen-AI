***ContextGen AI – RAG-Based Document Question Answering Bot***

ContextGen AI is a ***simple and lightweight RAG*** (Retrieval-Augmented Generation) chatbot that allows users to upload any document and ask questions directly from it — ***without using any external API***.

It works entirely locally using ***embeddings + similarity search***, making it ***fast, private, and easy to run***.

🚀 ***Features***

📄 ***Upload any document*** (PDF / TXT / etc.)

🤖 ***Asks questions and gets answers*** only from the uploaded document

🔍 ***Uses local embeddings*** (no API keys required)

⚡ ***Fast and privacy-focused***

🧠 ***Simple RAG pipeline*** (split → embed → retrieve → generate)

🖥️ ***Clean minimal UI*** (if using Streamlit)

🛠️ ***How It Works (RAG Pipeline)***

***Upload Document***
User uploads a file (PDF, text file, notes, etc.).

***Text Extraction***
The document is converted into raw text.

***Chunking***
The text is split into small meaningful chunks.

***Embeddings***
Each chunk is converted into vector embeddings locally.

***Similarity Search***
When a question is asked, the bot finds the most relevant chunks.

***Answer Generation***
The chatbot generates an answer only from the retrieved text, ensuring accuracy.

📥 ***Installation***
`git clone https://github.com/yourusername/contextgen-ai.git`
`cd contextgen-ai`
`pip install -r requirements.txt`

▶️ ***Run the App***

If using Streamlit:
`streamlit run app.py`

If it's a Python script:
`python app.py`

📘 ***Usage***

- Open the app
- Upload your document
- Ask any question related to the document
- Get instant contextual answers

***Example:***
Q: "What is the main objective mentioned in the document?"
A: Bot replies using only the uploaded content.

🧩 ***Tech Stack***
- Python
- LangChain (optional)
- Local Embeddings model
- Vector Store (FAISS / ChromaDB)
- Streamlit UI (if used)

🔒 ***Why No API?***

This project is designed to be:
- 💸 ***Cost-free*** (no API charges)
- 🔐 ***Privacy-safe*** (data never leaves your device)
- ⚡ ***Fast and lightweight***

🙌 Author
Aditya

Built with ❤️ Streamlit
