from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loadcsv import *
from faissRetrieval import *
from embedding import *
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("../knowledge_base.csv")
test_data = [
    {"User Input": "I feel worried", "Expected Response": "It's okay to feel worried. Try to focus on what you can control right now."},
    {"User Input": "I am so sad", "Expected Response": "I'm really sorry you're feeling this way. It might help to talk to someone you trust."},
]

def calculate_similarity(expected_response, faiss_response):
    expected_response_embed = model.encode([expected_response])
    faiss_response_embed = model.encode([faiss_response])

    expected_vs_faiss_sim = cosine_similarity(expected_response_embed, faiss_response_embed)[0][0]


    return expected_vs_faiss_sim

# Test the model
similarities = []
for test in test_data:
    user_input = test["User Input"]
    expected_response = test["Expected Response"]

    embed = embeddingModel([user_input])
    faissIndex = faiss.read_index("faiss_index")
    D, I = faissIndex.search(np.array(embed).astype('float32'), k=1)
    faiss_response = df.iloc[I[0][0]]['Response']

    faiss_similarity= calculate_similarity(expected_response, faiss_response)
    similarities.append(faiss_similarity)

    print(f"User Input: {user_input}")
    print(f"FAISS Response: {faiss_response}")
    print(f"Expected Response: {expected_response}")
    print(f"Cosine Similarity (FAISS vs Expected): {faiss_similarity:.2f}")
    print("--------")


average_similarity = sum(similarities) / len(similarities)
print(f"Average Semantic Similarity: {average_similarity:.2f}")
