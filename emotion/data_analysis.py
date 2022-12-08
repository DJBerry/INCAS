import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

pd.set_option('display.max_columns', None)

df = pd.read_pickle("annotate_emotion_phase_1b_sample_dataset.p")

print(df.sample(5))


