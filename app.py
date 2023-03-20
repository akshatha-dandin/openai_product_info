import os

import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
from flask import Flask, redirect, render_template, request, url_for

EMBEDDING_MODEL = "text-embedding-ada-002"

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

MAX_SECTION_LEN = 300
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))
print(f"Context separator contains {separator_len} tokens")
df = pd.read_csv('products_info_text.csv', encoding = "utf-8")
df = df.set_index(["title", "heading"])
print(f"{len(df)} rows in the data.")

def get_embedding(text: str):
    result = openai.Embedding.create(
      model="text-embedding-ada-002",
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }

document_embeddings = []
if not document_embeddings:
    print(f"Creating document embeddings")
    document_embeddings = compute_doc_embeddings(df)
    example_entry = list(document_embeddings.items())[0]
    print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        attribute = request.form["attribute"]
        prompt=construct_prompt(attribute, document_embeddings, df)
        print("===\n", prompt)

#Uncomment the below block to use davinci model
#         response = openai.Completion.create(
#                     model="text-davinci-003",
#                     prompt=prompt,
#                     max_tokens=150,
#                     temperature=0,
#                 )
#         return redirect(url_for("index", result=response.choices[0].text))
#davinci ends

#Uncomment the below block to use gpt-3.5-turbo model which has a lesser cost
        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
        return redirect(url_for("index", result=response['choices'][0]['message']['content']))
#gpt-3.5-turbo ends

    result = request.args.get("result")
    return render_template("index.html", result=result)


# def generate_prompt(animal, attribute):
# # Answer the question as truthfully as possible, and if you're unsure of the answer, say "Sorry, I don't know"
#     return """
#     Context: The Eddie Bauer "Guide Pro Pants" are best for hiking. Their material is a blend of Nylon and Spandex. They offer sun protection.They have an athletic fit.
#     Q:{}? A:""".format(attribute)

def vector_similarity(x: list, y: list):
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)[:5]

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "Sorry, I don't know."\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
