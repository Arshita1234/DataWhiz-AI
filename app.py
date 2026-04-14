import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma

# --- 1. PAGE SETUP & SESSION STATE ---
st.set_page_config(page_title="DataWhiz Pro", layout="wide", page_icon="🧠")
st.title("DataWhiz Pro: Persistent Multi-Doc AI")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.header("Control Panel")
api_key = st.sidebar.text_input("Enter Google API Key", type="password")
clear_db = st.sidebar.button("Clear Database & History")

if clear_db:
    # Logic to wipe the local database folder and chat
    if os.path.exists("./chroma_db"):
        import shutil
        shutil.rmtree("./chroma_db")
    st.session_state.messages = []
    st.rerun()

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # --- 3. MULTI-FILE UPLOADER ---
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    # Setup Embeddings Model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_document"
    )

    # Initialize or Load Persistent Chroma DB
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    if uploaded_files:
        new_files = False
        for uploaded_file in uploaded_files:
            # Check if we've already processed this file (simple check)
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Basic logic: If the DB is empty or we want to add, process it
            # (Note: In a production app, you'd check file hashes here)
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(pages)
            
            vectorstore.add_documents(docs)
            new_files = True
        
        if new_files:
            st.sidebar.success(f"{len(uploaded_files)} files processed and saved!")

    # --- 4. CHAT INTERFACE ---
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask about your documents..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analyzing..."):
            # RAG Logic
            relevant_docs = vectorstore.similarity_search(prompt, k=5)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.2)
            
            # Incorporating chat history into the prompt for context
            full_prompt = f"Context from PDFs:\n{context}\n\nQuestion: {prompt}"
            
            response = llm.invoke(full_prompt)
            
            # Handle the JSON/List output issue
            if isinstance(response.content, list):
                answer = response.content[0].get('text', str(response.content))
            else:
                answer = response.content

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Enter your API Key to start your workspace.")