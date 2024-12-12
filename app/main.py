import pandas as pd
import faiss
import streamlit as st
from loadcsv import *
from faissRetrieval import *
from embedding import *

# Load the dataframe
df = pd.read_csv("../knowledge_base.csv")

# Initialize the FAISS index
faissIndex = faiss.read_index("faiss_index")

# Check if 'messages' is already in session state, if not initialize it
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Define function to generate response
def genResponse(userInput):
    embed = embeddingModel([userInput])
    D, I = faissIndex.search(np.array(embed).astype('float32'), k=1)
    response = df.iloc[I[0][0]]

    res1 = {
        "role": 'user',
        "content": userInput
    }

    res2 = {
        "role": 'assistant',
        "content": response["Response"]
    }

    # Append new messages to session state
    st.session_state.messages.append(res1)
    st.session_state.messages.append(res2)
    print(st.session_state.messages)
# Display the app's title and caption
st.title("Mental Health Chat Companion")
st.caption("A compassionate companion for your mental well-being. Offering support, resources, and a listening ear, anytime you need it.")

# Display all chat messages
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        message1 = st.chat_message("user")
        message1.write(msg['content'])
    else:
        message2 = st.chat_message("assistant")
        message2.write(msg['content'])

# Chat input for the user
prompt = st.chat_input("Say something")
if prompt:
    genResponse(prompt)
    st.rerun()

