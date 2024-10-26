import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

# Initialize models
llm = ChatOllama(model="llama3.1", temperature=0.7)
embedding_model = OllamaEmbeddings(model="llama3.1")

# Load PDF documents
pdf_reader1 = PyPDFLoader("ttestfolder/Module 1 MSS PDF Notes - Kerala Notes (2).pdf")
pdf_reader2 = PyPDFLoader("ttestfolder/MSS M2 Ktunotes.in.pdf")
documents = pdf_reader1.load_and_split() + pdf_reader2.load_and_split()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200, length_function=len)
chunks = text_splitter.split_documents(documents)

# Create vector store
vector_store = Chroma(persist_directory="db", embedding_function=embedding_model)

# Setup chat prompt
chat_prompt = PromptTemplate.from_template(
    """
    You are an AI Teacher who is an EXPERT in teaching students.
    You have to read the documents from the students and teach them the contents provided in the
    PDF in an efficient and effective manner.
    You are also a friendly Chat Bot.
    Question: {question}
    Context: {context}
    Response:
    """
)

# Retrieve function
retriever = vector_store.as_retriever()

def handle_user_query(query):
    if not query:
        return "Please ask a question."

    docs = retriever.invoke(query)
    context = "\n".join(doc.content for doc in docs)  # Assuming docs have a 'content' attribute
    prompt = chat_prompt.format(question=query, context=context)

    # Generate response
    result = llm.invoke(prompt)
    print(result.content)
    return result.content

# Streamlit UI
st.title("Chat with Your PDF Documents")

# Session state to keep track of chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for chat in st.session_state.history:
    st.write(f"**User:** {chat['user']}")
    st.write(f"**Bot:** {chat['bot']}")

# User query input
user_question = st.text_input("Ask a question about your documents:")
if st.button("Send"):
    if user_question:
        response = handle_user_query(user_question)
        # Store in history
        st.session_state.history.append({"user": user_question, "bot": response})
        # Clear input box
        st.experimental_rerun()

st.write("Static PDFs are loaded from the specified folder.")
