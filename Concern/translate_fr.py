import os

CAT_DIR = ["ukraine", "russia", "refugees", "NATO", "energy", "economy", "economic sanctions", "defense"]

for concern in CAT_DIR:
    print(concern)
    en_files = os.listdir("Wiki_Content/{}/en".format(concern))
    ts_files = os.listdir("Wiki_Content/{}/ts_fr".format(concern))
    todo_files = list(set(en_files) - set(ts_files))
    print(todo_files)
    # break






