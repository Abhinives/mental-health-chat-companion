from loadcsv import *
from faissRetrieval import *
from embedding import *
# from responseGenerator import *
import faiss
import streamlit as st

df = pd.read_csv("../knowledge_base.csv")

# triggerPhrase = list(df['Trigger Phrase'])
# print(list(df['Trigger Phrase']))
# embeddings = embeddingModel(triggerPhrase)

# faiss = faissStore(embeddings)
# print(type(df))



# faissIndex = faiss.read_index("faiss_index")

# userInput = "I feel worried"

# embed = embeddingModel([userInput])

# D,I = faissIndex.search(np.array(embed).astype('float32'),k=1)
# response = df.iloc[I[0][0]]['Response']


# print(generate_response(userInput, response))


st.title("Mental Health Chat Companion")
st.caption("A compassionate companion for your mental well-being. Offering support, resources, and a listening ear, anytime you need it.")

message1 = st.chat_message("user")
message2 = st.chat_message("assistant")

message1.write("Hi")
message2.write("Hello")
prompt = st.chat_input("Say something")
if prompt:
    st.write("Sent")

