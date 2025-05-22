from utils.utils_retrieval import *
from utils.utils_generation import *

def main(question, lang):
    assert TOP_K <= NUM_SEARCHES, "NUM_SEARCHES must be equal or greater than TOP_K"

    start_time = time.time()

    print("> Searching on Google...")
    results = search_query(question, lang)
    # results = example_search_result(lang)

    print("> Embedding and ranking the results...")
    top_context = rank_searches(question, results)

    print("> Calling ChatGPT-4o...")
    response = generate_answer(question, top_context, lang)

    print("\n")
    print("-" * 70)
    print(f"> {ANSWER_GUIDER[lang][0]}: {response}")
    print("-" * 70)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"> Total time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    lang = input(f"Language ('tr' or 'en'): ")  # either tr or en
    lang = "tr" if lang != "en" else "en"
    question = input(f"{PROMPT_GUIDER[lang][3]}: ")
    main(question, lang)
