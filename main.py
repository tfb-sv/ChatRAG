import os
from utils.utils_retrieval import *
from utils.utils_generation import *

def main(question, query, conversation_history, lang):
    assert TOP_K <= NUM_SEARCHES, "NUM_SEARCHES must be equal or greater than TOP_K"

    start_time = time.time()

    if query:
        print("> Searching on Google...")
        results = search_query(query, lang)
        # results = example_search_result(lang)

        print("> Embedding and ranking the results...")
        top_context = rank_searches(query, results)
    else: top_context = None

    print("> Calling ChatGPT-4o...")
    response = generate_answer(
        question, top_context, conversation_history, lang
    )

    print("\n")
    print("-" * 70)
    print(f"> {PROMPT_GUIDER[lang][5]}: {response}")
    print("-" * 70)
    print("\n")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"> Total time: {elapsed:.2f} seconds")

    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": response})

    return conversation_history

def reset_history(lang):
    return [
        {"role": "system", "content": PROMPT_GUIDER[lang][0]},
        {"role": "user", "content": PROMPT_GUIDER[lang][1]}
    ]

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

if __name__ == "__main__":
    print(
"""
ChatRAG: A ChatGPT + Google-powered RAG chat system

    - Select a language: "tr" (Turkish) or "en" (English).
    - If you enter a query, ChatGPT replies using information fetched from Google.
    - If you don't enter a query, ChatGPT replies using only prior context.
    - Type "reset" as the question input to clear the chat history.
    - Type "exit" as the question input to end the session.\n
"""
)

    lang = input(f"Language: ")
    lang = "tr" if lang != "en" else "en"
    conversation_history = reset_history(lang)
    while True:
        query = input(f"\n{PROMPT_GUIDER[lang][6]}: ")
        question = input(f"\n{PROMPT_GUIDER[lang][4]}: ")
        if question.lower() == "exit":
            print("> Conversion was ended.")
            break
        elif question.lower() == "reset":
            conversation_history = reset_history(lang)
            clear_terminal()
            print("> Conversion history was reset.")
            continue
        query = None if query == "" else query
        conversation_history = main(
            question, query, conversation_history, lang
        )
