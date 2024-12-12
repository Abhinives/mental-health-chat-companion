import faiss
import numpy as np


def faissStore(embeddings):
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "faiss_index")
    return index
