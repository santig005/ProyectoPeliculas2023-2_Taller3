from django.shortcuts import render
from django.core.management.base import BaseCommand
from movie.models import Movie
import os
import numpy as np

import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

from dotenv import load_dotenv, find_dotenv
# Create your views here.
def recommendations(request):
    #Se lee del archivo .env la api key de openai
    description = request.GET.get('searchMovie')
    if(description):
        _ = load_dotenv('../openAI.env')
        openai.api_key  = os.environ['openAI_api_key']
        items = Movie.objects.all()
        emb_req = get_embedding(description,engine='text-embedding-ada-002')

        sim = []
        for i in range(len(items)):
            emb = items[i].emb
            emb = list(np.frombuffer(emb))
            sim.append(cosine_similarity(emb,emb_req))
        sim = np.array(sim)
        idx = np.argmax(sim)
        idx = int(idx)
        respuesta=[(items[idx])]

        return render(request, 'recommendations.html',{'resultado':respuesta,'description':description})
    else:
        return render(request, 'recommendations.html')