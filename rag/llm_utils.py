import os
import logging
from typing import List, Optional
import asyncio
from asyncio import Semaphore
from time import time, sleep
import numpy as np
import openai
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Set up OpenAI API configuration
openai.api_type = "azure"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://test-oai69420.openai.azure.com/"
openai.api_version = "2023-05-15"

# Ensure API key is set
if openai.api_key is None:
    logging.warning("openai.api_key is None")

# Initialize global counters
start = None
num_requests = 0

class TokenBucket:
    """
    Implements a token bucket rate limiter.

    Attributes
    ----------
    rate : int
        The number of requests per second this bucket allows.
    """
    def __init__(self, rate: int):
        self._rate = rate
        self._capacity = rate
        self._tokens = self._capacity
        self._last_refill = time()

    async def consume(self):
        """Consume a token. If no tokens are available, waits until they are refilled."""
        while self._tokens < 1:
            self._refill()
            await asyncio.sleep(1)  # Wait for token to refill
        self._tokens -= 1
        global num_requests
        num_requests += 1

    def _refill(self):
        """Refills tokens in the bucket based on time elapsed."""
        now = time()
        time_passed = now - self._last_refill
        refill_amount = time_passed * self._rate
        self._tokens = min(self._capacity, self._tokens + refill_amount)
        self._last_refill = now

MaybeTokenBucket = Optional[TokenBucket]

async def _embed_sentence(sentence: str, token_bucket: MaybeTokenBucket = None) -> List[float]:
    """
    Asynchronously embeds a sentence using OpenAI's API, respecting rate limits.

    Parameters
    ----------
    sentence : str
        The sentence to embed.
    token_bucket : MaybeTokenBucket, optional
        The token bucket to use for rate limiting.

    Returns
    -------
    List[float]
        The embedding of the sentence.
    """
    done = False
    while not done:
        try:
            if token_bucket is not None:
                await token_bucket.consume()
            response = await openai.Embedding.acreate(
                input=sentence,
                engine="embeddings"
            )
            embedding = response['data'][0]['embedding']
            done = True
        except Exception as e:
            if token_bucket is None:
                sleep(1)
            print(e)
    return embedding

def embed_sentence(sentence: str) -> List[float]:
    """Synchronously embeds a sentence using asyncio."""
    return asyncio.run(_embed_sentence(sentence))

async def _handle_sentence(sentence: str, token_bucket: TokenBucket, semaphore: Semaphore) -> List[float]:
    """
    Asynchronously handles the embedding of a sentence, including concurrency control.

    Parameters
    ----------
    sentence : str
        The sentence to embed.
    token_bucket : TokenBucket
        The token bucket to use for rate limiting.
    semaphore : Semaphore
        The semaphore to limit the number of concurrent tasks.

    Returns
    -------
    List[float]
        The embedding of the sentence.
    """
    async with semaphore:
        embedding = await _embed_sentence(sentence, token_bucket)
    global num_requests
    if num_requests % 1000 == 0:
        duration = time() - start
        duration_min = duration / 60
        print(f"{num_requests=}, {duration=:.2f} rate per min={num_requests / duration_min:.2f}")
    return embedding

def embed_sentences(sentences: List[str]) -> List[List[float]]:
    """
    Embeds multiple sentences concurrently.

    Parameters
    ----------
    sentences : List[str]
        The sentences to embed.

    Returns
    -------
    List[List[float]]
        A list of embeddings.
    """
    global start, num_requests
    start = time()
    num_requests = 0
    max_concurrent_tasks = 20
    azure_quota_per_minute = 1000
    azure_quota_per_second = azure_quota_per_minute // 60
    semaphore = Semaphore(max_concurrent_tasks)
    token_bucket = TokenBucket(azure_quota_per_second)
    async def gather_tasks():
        tasks = [_handle_sentence(sentence, token_bucket, semaphore) for sentence in sentences]
        return await asyncio.gather(*tasks)
    return asyncio.run(gather_tasks())

async def _call_chat(system_prompt: str, user_prompt: str) -> str:
    """
    Asynchronously calls the chat completion API with specified system and user prompts.

    Parameters
    ----------
    system_prompt : str
        The system prompt to send to the chat model.
    user_prompt : str
        The user prompt to send to the chat model.

    Returns
    -------
    str
        The completed response from the chat model.
    """
    response = await openai.ChatCompletion.acreate(
        engine="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    completion = response.choices[0].message.content
    return completion

def call_chat(system_prompt: str, user_prompt: str) -> str:
    """
    Synchronously calls the chat completion API.

    Parameters
    ----------
    system_prompt : str
    user_prompt : str

    Returns
    -------
    str
        The response from the chat model.
    """
    return asyncio.run(_call_chat(system_prompt, user_prompt))

def cosine_similarity(query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
    """
    Calculates the cosine similarity between a query embedding and a set of document embeddings.

    Parameters
    ----------
    query_embedding : np.ndarray
        A single query embedding vector.
    document_embeddings : np.ndarray
        An array of document embedding vectors.

    Returns
    -------
    np.ndarray
        Cosine similarities between the query and each document.
    """
    return document_embeddings @ query_embedding

def argknn(query_embedding: np.ndarray, document_embeddings: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Finds the top k nearest neighbors using cosine similarity.

    Parameters
    ----------
    query_embedding : np.ndarray
        The query embedding vector.
    document_embeddings : np.ndarray
        An array of document embedding vectors.
    k : int
        The number of nearest neighbors to retrieve.

    Returns
    -------
    np.ndarray
        Indices of the top k nearest documents.
    """
    dots = cosine_similarity(query_embedding, document_embeddings)
    pivot = len(dots) - k
    good_idxes = np.argpartition(dots, pivot)[-k:]
    sorted_good_idxes = good_idxes[np.argsort(good_idxes)[::-1]]
    return sorted_good_idxes

def test_embeddings():
    """
    Tests embedding functionality with a single sentence and batch of sentences.
    """
    sentence = "George Biden was the 46th President of the United States"
    embedding = embed_sentence(sentence)
    print(f"Embedding truncated to 5 elements: {embedding[:5]}")
    sentences = [sentence, "Old MacDonald had a farm EIEIEO", "The quick brown fox jumped over the lazy dog"]
    embeddings = embed_sentences(sentences)
    print('Batch Embeddings:')
    for s, e in zip(sentences, embeddings):
        print(f"Sentence: {s}, Embedding: {e[:5]}")

def test_cosine_similarity():
    """
    Tests the cosine similarity and argknn functions by comparing a query sentence to a list of document sentences.
    """
    documents = ["George Biden was the 46th President of the United States", "Old MacDonald had a farm EIEIEO", "The quick brown fox jumped over the lazy dog"]
    query = "And on that farm there was a dog"
    query_embedding = embed_sentence(query)
    document_embeddings = np.array(embed_sentences(documents))
    cosine_sims = cosine_similarity(query_embedding, document_embeddings)
    k = 2
    top_k_idxes = argknn(query_embedding, document_embeddings, k)
    top_k_docs = [documents[i] for i in top_k_idxes]
    print(f"Query: {query}, Top {k} documents: {top_k_docs}")

if __name__ == '__main__':
    test_embeddings()
    test_cosine_similarity()