from sentence_transformers import SentenceTransformer


def embeddingModel(triggerPhrases):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(triggerPhrases)
    embeddings = model.encode(triggerPhrases, show_progess_bar=True)

    return embeddings