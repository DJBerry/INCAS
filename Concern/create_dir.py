import os

if __name__ == "__main__":

    concerns = [
        "ukraine"
        "russia",
        "NATO",
        "refugees",
        "defense",
        "economy",
        "economic sanctions",
        "energy"
    ]

    dirs = []

    for concern in concerns:
        dirs.append("{}/en".format(concern))
        dirs.append("{}/fr".format(concern))

    for dir in dirs:
        try:
            os.makedirs(dir)
            print("created: '{}'".format(dir))
        except FileExistsError:
            print("already exists: '{}'".format(dir))