import os
from utils.utils_retrieval import *
from utils.utils_generation import *

def main(question, conversation_history, lang):
    assert TOP_K <= NUM_SEARCHES, "NUM_SEARCHES must be equal or greater than TOP_K"

    start_time = time.time()

    print("> Searching on Google...")
    results = search_query(question, lang)
    # results = example_search_result(lang)

    print("> Embedding and ranking the results...")
    top_context = rank_searches(question, results)

    print("> Calling ChatGPT-4o...")
    response = generate_answer(
        conversation_history, question, top_context, lang
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
    lang = input(f"Language ('tr' or 'en'): ")  # either tr or en
    lang = "tr" if lang != "en" else "en"
    conversation_history = reset_history(lang)
    while True:
        question = input(f"\n{PROMPT_GUIDER[lang][4]}: \n")
        if question.lower() == "exit": break
        elif question.lower() == "restart":
            conversation_history = reset_history(lang)
            clear_terminal()
            continue
        conversation_history = main(question, conversation_history, lang)
