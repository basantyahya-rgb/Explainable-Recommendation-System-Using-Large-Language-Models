import gradio as gr
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from chromadb.config import Settings
from chromadb import Client
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser


#  Load books.csv

df = pd.read_csv(
    r"D:\Cybersecurity\Queen's\ML\Project\RAG\books.csv",
    sep=";",
    encoding="latin-1",
    engine="python",
    on_bad_lines="skip"
)

print(f"Total books loaded: {len(df)}")
print("Columns:", df.columns)

# ⚠️ Limit dataset for faster testing (IMPORTANT)
df = df.head(5000)

documents = []

for idx, row in df.iterrows():
    content = f"""
    Title: {row.get('Book-Title', '')}
    Author: {row.get('Book-Author', '')}
    Year: {row.get('Year-Of-Publication', '')}
    Publisher: {row.get('Publisher', '')}
    """

    documents.append(
        Document(
            page_content=content,
            metadata={"isbn": row.get("ISBN", "")}
        )
    )

print(f"Total documents created: {len(documents)}")
print("Sample document:\n", documents[0].page_content)


# Create Embeddings (Ollama)


embedding_function = OllamaEmbeddings(model="nomic-embed-text")


def generate_embedding(doc):
    return embedding_function.embed_query(doc.page_content)


print("Generating embeddings...")

with ThreadPoolExecutor() as executor:
    embeddings = list(executor.map(generate_embedding, documents))

print(f"Total embeddings generated: {len(embeddings)}")



#  Store in Chroma Vector Database

client = Client(Settings(anonymized_telemetry=False))

# Delete old collection if exists
try:
    client.delete_collection(name="books_collection")
except:
    pass

collection = client.create_collection(name="books_collection")

print("Storing documents in Chroma...")

for idx, doc in enumerate(documents):
    collection.add(
        documents=[doc.page_content],
        metadatas=[doc.metadata],
        embeddings=[embeddings[idx]],
        ids=[str(idx)]
    )

# Create retriever
retriever = Chroma(
    collection_name="books_collection",
    client=client,
    embedding_function=embedding_function
).as_retriever(search_kwargs={"k": 3})


# Initialize LLM (Ollama LLaMA3)

llm = ChatOllama(
    model="llama3",
    temperature=0.5,
)

parser = StrOutputParser()


# RAG Recommendation Function

def recommend_books(user_query):
    retrieved_docs = retriever.invoke(user_query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    formatted_prompt = f"""
    User preference: {user_query}

    Recommended books:
    {context}

    Explain clearly why these books match the user's interest.
    Provide a short personalized explanation.
    """

    response = llm.invoke(formatted_prompt)
    answer = parser.invoke(response)

    return answer


#UI part

def recommend_books_formatted(user_query):
    retrieved_docs = retriever.invoke(user_query)

    books = []
    for doc in retrieved_docs:
        books.append(doc.page_content)

    context = "\n\n".join(books)

    formatted_prompt = f"""
    User preference: {user_query}

    Recommended books:
    {context}

    Respond strictly in this format:

    Title: ...
    Author: ...
    Reason: ...

    Separate each book with ---
    Keep it concise and professional.
    """

    response = llm.invoke(formatted_prompt)
    answer = parser.invoke(response)

    return answer


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

body {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #f3e7e9, #e3eeff);
}

#header {
    text-align:center;
    margin-bottom:25px;
}

#header h1 {
    font-weight:600;
    color:#2c3e50;
}

#header p {
    color:#6c757d;
}

.guide-box {
    background:white;
    padding:18px;
    border-radius:14px;
    margin-bottom:20px;
    box-shadow:0 8px 20px rgba(0,0,0,0.06);
}

.result-container {
    max-height: 500px;
    overflow-y: auto;
}

.result-card-wrapper {
    padding:3px;
    border-radius:16px;
    margin-bottom:18px;
    background: linear-gradient(90deg, #a18cd1, #fbc2eb);
}

.result-box {
    background:white;
    padding:22px;
    border-radius:14px;
    transition: all 0.25s ease;
}

.result-box:hover {
    transform: translateY(-4px);
    box-shadow:0 12px 25px rgba(0,0,0,0.08);
}

.result-box h3 {
    margin-bottom:8px;
    font-weight:600;
    color:#34495e;
}

.result-box p {
    margin:4px 0;
    color:#555;
    font-size:14px;
}

button {
    font-family:'Inter', sans-serif !important;
    font-weight:600;
}
"""

with gr.Blocks(css=custom_css) as demo:

    gr.HTML("""
        <div id="header">
            <h1>AI Book Recommendation System</h1>
            <p>Explainable Recommendations Powered by LLaMA3 + Semantic Retrieval</p>
        </div>
    """)

    gr.HTML("""
        <div class="guide-box">
            <b>How it works:</b>
            <ul>
                <li>Describe your interest.</li>
                <li>The system retrieves semantically relevant books.</li>
                <li>AI generates clear explanations for each recommendation.</li>
                <li>Latest results appear at the top.</li>
            </ul>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            user_input = gr.Textbox(
                label="Describe your book preference",
                placeholder="Example: Beginner-friendly cybersecurity books",
                lines=2
            )

            recommend_btn = gr.Button("Generate Recommendations", variant="primary")
            clear_btn = gr.Button("Clear History")

        with gr.Column(scale=5):
            output_area = gr.HTML(elem_classes="result-container")

    history = []

    def handle_recommendation(query):
        if not query.strip():
            return "<div class='result-card-wrapper'><div class='result-box'><b>Please enter your interest.</b></div></div>"

        raw_result = recommend_books_formatted(query)
        books = raw_result.split("---")

        formatted_cards = ""

        for book in books:
            lines = book.strip().split("\n")
            title = ""
            author = ""
            reason = ""

            for line in lines:
                if line.lower().startswith("title"):
                    title = line.split(":",1)[1].strip()
                elif line.lower().startswith("author"):
                    author = line.split(":",1)[1].strip()
                elif line.lower().startswith("reason"):
                    reason = line.split(":",1)[1].strip()

            formatted_cards += f"""
            <div class='result-card-wrapper'>
                <div class='result-box'>
                    <h3>{title}</h3>
                    <p><b>Author:</b> {author}</p>
                    <p><b>Why Recommended:</b> {reason}</p>
                </div>
            </div>
            """

        # newest result first
        history.insert(0, formatted_cards)

        return "".join(history)

    recommend_btn.click(
        handle_recommendation,
        inputs=user_input,
        outputs=output_area
    )

    clear_btn.click(
        lambda: ("", ""),
        outputs=[user_input, output_area]
    )

demo.launch(share=True)