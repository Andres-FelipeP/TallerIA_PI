from dotenv import load_dotenv, find_dotenv
import json
import os
from openai import OpenAI
import numpy as np
from .models import Movie


def buscador_ia(text_promt):
    _ = load_dotenv('../api_keys.env')
    client = OpenAI(
        api_key=os.environ.get('openai_apikey'),
    )

    def get_embedding(text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    emb_user_text_promt = get_embedding(text_promt)

    movies = Movie.objects.all()
    similar_movies = []

    for movie in movies:
        rec_emb = np.frombuffer(movie.emb, dtype=np.float32)  # Asegúrate de que el dtype coincida

        if cosine_similarity(emb_user_text_promt, rec_emb) > 0.4:
            similar_movies.append(movie)

    return similar_movies

