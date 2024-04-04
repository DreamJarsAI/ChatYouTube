# Create an app to chat with a YouTube video transcript


# Import modules
import os
from apikey import apikey
import streamlit as st
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import YoutubeLoader


#  Set API key
os.environ['OPENAI_API_KEY'] = apikey


# Define a function to clear the chat history from the session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


# Set title
st.title("Chat with a document")


# Ask the user to provide a YouTube video URL
youtube_url = st.text_input("Enter a YouTube video URL")


# Check if the user provided a YouTube video URL
if youtube_url:
    # Load the YouTube video transcript as a text document
    loader = YoutubeLoader.from_youtube_url(youtube_url)
    documents = loader.load()

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split the document
    chunks = text_splitter.split_documents(documents)

    # Embed the chunks
    embeddings = OpenAIEmbeddings()

    # Initialize the vectorstore
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # Initialize the LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # Initialize the retriever
    retriever = vectorstore.as_retriever()

    # Define the conversational retrieval chain
    crc = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

    # Set the conversational retrieval chain in the session state
    st.session_state.crc = crc


# Get question from the user
question = st.text_input("Ask a question about the YouTube video:")


# Get the answer
if question:
    # Add the conversational retrieval chain to the session state.
    # Note: this is necessary because question is a new user input in the app, which is separated from the file upload button.
    if 'crc' in st.session_state:
        crc = st.session_state.crc

        # Save chat history in the session state
        if 'history' not in st.session_state:
            st.session_state.history = []
    
        # Run the conversational retrieval chain by passing the question and chat history
        response = crc.run({
            'question': question, 
            'chat_history': st.session_state.history})
    
        # Add the question and answer to the chat history
        st.session_state.history.append((question, response))
        st.write(response)

        # Display the answer
        for prompts in st.session_state.history:
            st.write("Question: " + prompts[0])
            st.write("Answer: " + prompts[1])