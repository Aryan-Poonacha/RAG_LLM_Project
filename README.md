# RAG LLM Info Bot

## Table of Contents
1. Introduction
2. Installation
3. Datasets 
4. Usage
5. Vectorization Methods
6. Chat Models
7. Workflow

## Introduction
The RAG LLM Info Bot is a conversational AI that uses a combination of vectorization methods and chat models to answer user queries based on the information extracted from uploaded PDF documents.

## Installation
Following are the steps to run the StreamLit Application: 


**1. Create a new venv and enter it:** 
**2. Install python package requirements:** 
```
pip install -r requirements.txt 
```
**4. Add OpenAI API Key**
```
Rename the env.example file to .env and add your OpenAI API key
```
**5. Run the application**
```
streamlit run app.py
```

## Datasets
Shakespeare plays and works of literature in clean PDF formats that have been Verified to read well with the used methods are loaded into the application once the streamlit application is launched. These are sourced from https://www.folger.edu/explore/shakespeares-works/download/ Folger Shakespeare Library, 201 East Capitol Street, SE, Washington, DC 20003.

## Usage
Run the streamlit app to start loading the plays into memory and start chatting after to ask questions about Shakespeare plays!

## Vectorization Methods
The application uses two different vectorization methods to convert text into numerical representations (vectors):

1. **BERT**: The BERT (Bidirectional Encoder Representations from Transformers) model is used to convert chunks of text into vectors. This is done in the `get_vectorstore_bert` function.

2. **SentenceTransformer**: The SentenceTransformer model is used to convert chunks of text into vectors. This is done in the `get_vectorstore` function.

## Chat Models
The application supports two different chat models:

1. **GPT-2**: The GPT-2 (Generative Pretrained Transformer 2) model can be used as the chat model. This is done in the `get_conversation_chain` function.

2. **GPT-3.5-turbo**: The GPT-3.5-turbo model can be used as the chat model. This is done in the `get_conversation_chain` function.

## Workflow
The application follows the following workflow:

1. **PDF Extraction**: The application extracts text from uploaded PDF documents using the `extract_text` function.

2. **Chunk Creation**: The extracted text is split into chunks using the `get_chunks` function.

3. **Vectorization**: The chunks of text are converted into vectors using the `get_vectorstore_bert` or `get_vectorstore` function.

4. **Memory Loading**: The vectors are loaded into memory.

5. **Query Vectorization**: The user's query is vectorized using the same vectorization method as the chunks.

6. **Retrieval**: The most similar chunk to the user's query is retrieved using cosine similarity in the `retrieve_most_similar_chunk_bert` or `retrieve_most_similar_chunk` function.

7. **Prompt Creation**: The retrieved chunk is used to create a prompt for the chat model in the `MyConversationalRetrievalChain` class.

8. **Response Generation**: The chat model generates a response based on the prompt in the `MyConversationalRetrievalChain` class.

9. **Output**: The response from the chat model is returned to the user.
