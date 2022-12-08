import numpy as np
import pandas as pd
import metachange
from datetime import datetime
from tqdm import tqdm
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

pd.set_option('display.max_columns', None)

X = np.array([[0,1]]*500 + [[1,0]]*500 + [[2,0]]*500 + [[2,1]]*500)
t = np.arange(2000)*1./2000

clf_rf = RandomForestClassifier(max_depth=32, criterion="entropy", random_state=0)
res_multi, res_multi_result = metachange.change_point_tree(X, t, clf_rf, min_range=0.10)

## define a funciton which generates node text
def make_node_text(data):
    t_left = data["t_left"]
    t_right = data["t_right"]

    if "t0" in data:
        header = f't_0 = {data["t0"]:.4f}\n alpha = {data["alpha"]:.4f}'
    else:
        header = "Leaf"
    return f"{header}\nRange:{t_left:.4f}-{t_right:.4f}"

tree = metachange.show_tree(res_multi, make_node_text, fname="kaitest2.pdf")
u = tree.unflatten(stagger=2)
u.render("kaichen2")

