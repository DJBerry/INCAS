import wikipediaapi
import wikipedia
from tqdm import tqdm

CAT_DIR = "defense"
# PAGE_NAME = "2022 Russian invasion of Ukraine"
CONCERN = "2022 defense"
# LANGEUAGE = "fr" "en"
LANGEUAGES = ["en", "fr"]

for LANGEUAGE in LANGEUAGES:
    print(LANGEUAGE)
    # get titles of top relevant articles
    wikipedia.set_lang(LANGEUAGE)
    top_relevant_articles = wikipedia.search(CONCERN, results=40)
    # print(top_relevant_articles)
    print("num of top relevant articles: {}".format(len(top_relevant_articles)))

    # initiate api
    wiki_wiki = wikipediaapi.Wikipedia(
        language=LANGEUAGE,
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

    # get full text based on title
    num_right_file = 0
    for article in tqdm(top_relevant_articles):
        p_wiki = wiki_wiki.page(article)  # Title of wikipedia page
        if p_wiki.exists():
            with open('Wiki_Content/{}/{}/{}.txt'.format(CAT_DIR, LANGEUAGE, article), 'w',
                      encoding='utf-8') as myfile:
                myfile.write(p_wiki.text)
            num_right_file += 1
        else:
            continue
    print("save {} files".format(num_right_file))