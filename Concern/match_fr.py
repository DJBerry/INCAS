import wikipediaapi
import wikipedia
from tqdm import tqdm
import os

CAT_DIR = ["ukraine", "russia", "refugees", "NATO", "energy", "economy", "economic sanctions", "defense"]
# PAGE_NAME = "2022 Russian invasion of Ukraine"
TITLE = "2022 Russian invasion of Ukraine"
# LANGEUAGE = "fr" "en"
LANGEUAGE = "en"

# print(LANGEUAGE)
# get titles of top relevant articles
wikipedia.set_lang(LANGEUAGE)
top_relevant_articles = wikipedia.search(TITLE, results=1)
print(top_relevant_articles)
# print("num of top relevant articles: {}".format(len(top_relevant_articles)))

# initiate api
wiki_wiki = wikipediaapi.Wikipedia(
    language=LANGEUAGE,
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

# def print_langlinks(page):
#     langlinks = page.langlinks
#     for k in sorted(langlinks.keys()):
#         v = langlinks[k]
#         print("%s: %s - %s: %s" % (k, v.language, v.title, v.fullurl))

# get full text based on title
num_right_file = 0
for concern in CAT_DIR:
    print(concern)
    files = os.listdir("Wiki_Content/{}/en".format(concern))
    # print(files)

    translate_fr = 0
    for file in files:
        article = file.strip(".txt")
        # print(article)
        try:
            page_py = wiki_wiki.page(article)
            page_py_fr = page_py.langlinks['fr']
            with open('Wiki_Content/{}/ts_fr/{}.txt'.format(concern, article), 'w', encoding='utf-8') as myfile:
                myfile.write(page_py_fr.text)
            translate_fr += 1
        # except wikipedia.exceptions.PageError:
        except:
            # print("fr not exit")
            continue
    num_en = len(files)
    # num_fr = len(titles_fr)
    num_fr = translate_fr
    ratio_fr = num_fr/num_en
    print("============= match result =============")
    print(num_en)
    print(num_fr)
    print(ratio_fr)
    print("========================================")

    # for article_fr in titles_fr:
    #     p_wiki = wiki_wiki.page(article_fr)  # Title of wikipedia page
    #     if p_wiki.exists():
    #         with open('Wiki_Content/{}/{}/{}.txt'.format(concern, LANGEUAGE, article_fr), 'w',
    #                   encoding='utf-8') as myfile:
    #             myfile.write(p_wiki.text)
    #         num_right_file += 1
    #     else:
    #         continue
    # print("save {} files".format(num_right_file))


# # get full text based on title
# num_right_file = 0
# for concern in CAT_DIR:
#     print(concern)
#     files = os.listdir("Wiki_Content/{}/en".format(concern))
#     # print(files)
#
#     titles_fr = []
#     for file in files:
#         article = file.strip(".txt")
#         # print(article)
#         try:
#             wikipedia.set_lang(LANGEUAGE)
#             article_fr = wikipedia.page(article)
#             article_title_fr = article_fr.title
#             # print(article_title_fr)
#             titles_fr.append(article_title_fr)
#         # except wikipedia.exceptions.PageError:
#         except:
#             # print("fr not exit")
#             continue
#     num_en = len(files)
#     num_fr = len(titles_fr)
#     ratio_fr = num_fr/num_en
#     print("============= match result =============")
#     print(num_en)
#     print(num_fr)
#     print(ratio_fr)
#     print("========================================")
#
#     for article_fr in titles_fr:
#         p_wiki = wiki_wiki.page(article_fr)  # Title of wikipedia page
#         if p_wiki.exists():
#             with open('Wiki_Content/{}/{}/{}.txt'.format(concern, LANGEUAGE, article_fr), 'w',
#                       encoding='utf-8') as myfile:
#                 myfile.write(p_wiki.text)
#             num_right_file += 1
#         else:
#             continue
#     print("save {} files".format(num_right_file))