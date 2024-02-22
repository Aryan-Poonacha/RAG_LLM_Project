import streamlit as st
from dotenv import load_dotenv

from html_chatbot_template import css, bot_template, user_template

from PyPDF2 import PdfReader

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from openai import ChatCompletion

import os
import torch
import numpy as np

class MyConversationalRetrievalChain:
    def __init__(self, model, tokenizer, vector_store, memory):
        self.model = model
        self.tokenizer = tokenizer
        self.vector_store = vector_store
        self.memory = memory

    def __call__(self, inputs):
        question = inputs['question']
        context = inputs.get('context', '')

        # Add the additional prompt
        prompt = f"Here is some relevant contextual information retrieved from the database: {context} Please use any relevant information, if available, in it to answer the following query: {question}"

        # Tokenize the inputs
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        # Generate a response
        chat_history_ids = self.model.generate(input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)

        # Decode the response
        response = self.tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Print the prompt sent to the language model
        print(f"Prompt to LLM: {prompt}")

        # Print the response from the language model
        print(f"Response from LLM: {response}")

        with open('latestpromptandresponse.txt', 'w') as f:
            f.write(f"Prompt to LLM: {prompt}\nResponse from LLM: {response}")

        return {'response': response}

    
class ConversationBufferMemory:
    def __init__(self, memory_key="chat_history", return_messages=True):
        self.memory_key = memory_key
        self.return_messages = return_messages

    def __call__(self, chat_state):
        if self.memory_key not in chat_state:
            chat_state[self.memory_key] = []
        if self.return_messages:
            return chat_state[self.memory_key]


# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model and tokenizer
vectorization_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
vectorization_model = AutoModel.from_pretrained('bert-base-uncased')

def extract_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def get_chunks(text, separator="\n", chunk_size=512, chunk_overlap=64):
    """
    Function to get the chunks of text from the raw text

    Args:
        text (str): The raw text from the PDF file

    Returns:
        chunks (list): The list of chunks of text
    """
    # Split the text by new line
    lines = text.split(separator)

    # Initialize the chunks list
    chunks = []

    # Iterate over the lines
    for line in lines:
        # Get the length of the line
        length = len(line)

        # If the length of the line is less than the chunk size, add the whole line to the chunks
        if length <= chunk_size:
            chunks.append(line)
        else:
            # If the length of the line is greater than the chunk size, split the line into chunks
            for i in range(0, length, chunk_size - chunk_overlap):
                chunk = line[i:i + chunk_size]
                chunks.append(chunk)

    return chunks


def get_vectorstore(chunks):
    """
    Function to create a vector store for the chunks of text to store the embeddings

    Args:
        chunks (list): The list of chunks of text

    Returns:
        vector_store (dict): The vector store for the chunks of text
    """
    # Initialize the vector store
    vector_store = {}

    # Iterate over the chunks
    for chunk in chunks:
        # Tokenize the chunk
        inputs = vectorization_tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)

        # Get the embeddings for the chunk
        with torch.no_grad():
            outputs = vectorization_model(**inputs)

        # Get the embeddings of the [CLS] token (first token)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        # Normalize the embeddings
        embeddings = normalize(embeddings)

        # Add the embeddings to the vector store
        vector_store[chunk] = embeddings

    return vector_store

def retrieve_most_similar_chunk(query, vector_store):
    """
    Function to retrieve the most similar chunk for a given query

    Args:
        query (str): The user query
        vector_store (dict): The vector store for the chunks of text

    Returns:
        most_similar_chunk (str): The most similar chunk for the given query
    """
    # Tokenize the query
    inputs = vectorization_tokenizer(query, return_tensors='pt', truncation=True, padding=True)

    # Get the embeddings for the query
    with torch.no_grad():
        outputs = vectorization_model(**inputs)

    # Get the embeddings of the [CLS] token (first token)
    query_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    # Normalize the query embeddings
    query_embeddings = normalize(query_embeddings)

    # Initialize the maximum similarity and the most similar chunk
    max_similarity = -1
    most_similar_chunk = None

    # Iterate over the chunks in the vector store
    for chunk, embeddings in vector_store.items():
        # Calculate the cosine similarity between the query embeddings and the chunk embeddings
        similarity = cosine_similarity(query_embeddings, embeddings)

        # If the similarity is higher than the maximum similarity, update the maximum similarity and the most similar chunk
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_chunk = chunk

    # Print the vectorized query
    print(f"Vectorized Query: {query_embeddings}")

    # Print the most similar chunk
    print(f"Most Similar Chunk: {most_similar_chunk}")

    with open('query_log.txt', 'w') as f:
        f.write(f"Vectorized Query: {query_embeddings}\nMost Similar Chunk: {most_similar_chunk}")

    return most_similar_chunk

def get_conversation_chain(vector_store):
    """
    Function to create a conversation chain for the chat model

    Args:
        vector_store (dict): The vector store for the chunks of text
    
    Returns:
        conversation_chain (MyConversationalRetrievalChain): The conversation chain for the chat model
    """
    # Get the model type from the environment variables
    model_type = os.getenv('MODEL_TYPE')

    if model_type == 'gpt2':
        # Initialize the chat model using local GPT-2 model
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif model_type == 'gpt-3.5-turbo':
        # Initialize the chat model using OpenAI API
        model = ChatCompletion.create(model="gpt-3.5-turbo", api_key=os.getenv('OPENAI_API_KEY'))
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Initialize the chat memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create a conversation chain for the chat model
    conversation_chain = MyConversationalRetrievalChain(
        model=model, # Chat model
        tokenizer=tokenizer, # Tokenizer
        vector_store=vector_store, # Vector store
        memory=memory, # Chat memory
    )

    return conversation_chain


def generate_response(question):
    """
    Function to generate a response for the user query using the chat model

    Args:
        question (str): The user query

    Returns:
        response (str): The response from the chat model
    """
    # Retrieve the most similar chunk for the user query
    context = retrieve_most_similar_chunk(question, st.session_state.vector_store)

    # Get the response from the chat model for the user query
    response = st.session_state.conversations({'question': question, 'context': context})

    # Add the response to the UI
    st.write(bot_template.replace(
        "{{MSG}}", response['response']), unsafe_allow_html=True)
    
    # Print the response added to the UI
    print(f"Response added to UI: {response['response']}")
    with open('ui_log.txt', 'w') as f:
        f.write(f"Response added to UI: {response['response']}")


## Landing page UI
def run_UI():
    """
    The main UI function to display the UI for the webapp

    Args:
        None

    Returns:
        None
    """

    # Load the environment variables (API keys)
    load_dotenv()

    # Set the page tab title
    st.set_page_config(page_title="DocuMentor", page_icon="🤖", layout="wide")

    # Add the custom CSS to the UI
    st.write(css, unsafe_allow_html=True)

    # Initialize the session state variables to store the conversations and chat history
    if "conversations" not in st.session_state:
        st.session_state.conversations = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vector_store" not in st.session_state: 
        st.session_state.vector_store = None  

    # Set the page title
    st.header("DocuMentor: Conversations with Your Data 🤖")

    # Input text box for user query
    user_question = st.text_input("Upload your data and ask me anything?")

    # Check if the user has entered a query/prompt
    if user_question:
        # Call the function to generate the response
        generate_response(user_question)

    # Sidebar menu
    with st.sidebar:
        st.subheader("Document Uploader")

        # Document uploader
        pdf_files = st.file_uploader("Upload a document you want to chat with", type="pdf", key="upload", accept_multiple_files=True)

        # Process the document after the user clicks the button
        if st.button("Start Chatting ✨"):
            # Add a progress spinner
            with st.spinner("Processing"):
                # Convert the PDF to raw text
                raw_text = extract_text(pdf_files)
                print((f"Raw Text: {raw_text}") )
                        
                # Get the chunks of text
                chunks = get_chunks(raw_text)
                print(f"Chunks: {chunks}")
                        
                # Create a vector store for the chunks of text
                st.session_state.vector_store = get_vectorstore(chunks)
                print(f"Vector Store: {st.session_state.vector_store}")  

                with open('log.txt', 'w') as f:
                    f.write(f"Raw Text: {raw_text}\nChunks: {chunks}\nVector Store: {st.session_state.vector_store}")

                # Create a conversation chain for the chat model
                st.session_state.conversations = get_conversation_chain(st.session_state.vector_store) 

# Application entry point
if __name__ == "__main__":
    # Run the UI
    run_UI()
