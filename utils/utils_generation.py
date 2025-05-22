import time
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

OPENAI_TOKEN = open("utils/credentials_openai.env", encoding="utf-8").read().strip()

EMBED_MODEL = "text-embedding-3-small"
OPENAI_MODEL = "gpt-4o"

PROMPT_GUIDER = {
    "tr": [
        "Sen yardımsever, bilge ve kibar bir kişisel asistansın.",
        "Şu ana dek yaptığımız konuşmalar:",
        "Lütfen aşağıda verilen bilgilere dayanarak soruyu kısaca yanıtla.",
        "Bilgi",
        "Soru",
        "Cevap",
        "Arama Sorgusu"
    ],
    "en": [
        "You're a helpful, wise, and polite personal assistant.",
        "The conversation we've had so far:",
        "Please answer the question briefly based on the information given below.",
        "Information",
        "Question",
        "Answer",
        "Search Query"
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
    print(f"    > Embedding time: {elapsed:.2f} seconds")

    return top_context

def generate_answer(
        question, context, conversation_history, lang
    ):
    start_time = time.time()

    if context:
        prompt = (
            f"{PROMPT_GUIDER[lang][2]}\n\n"
            f"{PROMPT_GUIDER[lang][3]}:\n{context}\n\n"
            f"{PROMPT_GUIDER[lang][4]}: {question}"
        )
    else: prompt = question

    messages = conversation_history + [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    result = response.choices[0].message.content.strip()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"    > Answer generation time: {elapsed:.2f} saniye")

    return result
