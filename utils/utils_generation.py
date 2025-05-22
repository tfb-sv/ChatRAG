import time
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

OPENAI_TOKEN = open("utils/credentials_openai.env", encoding="utf-8").read().strip()

EMBED_MODEL = "text-embedding-3-small"
OPENAI_MODEL = "gpt-4o"

PROMPT_GUIDER = {
    "tr": [
        "Aşağıda verilen bilgiye dayanarak soruyu kısaca yanıtla.",
        "Bilgi",
        "Soru",
        "Cevap"
    ],
    "en": [
        "Answer the question briefly based on the information given below.",
        "Information",
        "Question",
        "Answer"
    ]
}

TOP_K = 5
MAX_TOKENS = 200
TEMPERATURE = 0.5

client = OpenAI(api_key=OPENAI_TOKEN)

def rank_searches(query, searches):
    start_time = time.time()

    texts = [query] + searches

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    embeddings = [item.embedding for item in response.data]

    query_vec = embeddings[0]
    passage_vecs = embeddings[1:]

    scores = cosine_similarity([query_vec], passage_vecs)[0]
    ranked = sorted(zip(searches, scores), key=lambda x: x[1], reverse=True)

    top_context = "\n".join([r[0] for r in ranked[:TOP_K]])

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\t> Embedding time: {elapsed:.2f} seconds")

    return top_context

def generate_answer(question, context, lang="tr"):
    start_time = time.time()

    prompt = (
        f"{PROMPT_GUIDER[lang][0]}\n\n"
        f"{PROMPT_GUIDER[lang][1]}:\n{context}\n\n"
        f"{PROMPT_GUIDER[lang][2]}: {question}"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    result = response.choices[0].message.content.strip()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\t> Answer generation time: {elapsed:.2f} saniye")

    return result
