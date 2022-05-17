import os
import streamlit as st
import numpy as np
from PIL import  Image

from multipage import MultiPage
from pages import image_modification, lidar_modification, home, results

import random
import numpy
import torch

SEED_VALUES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
rep_num = int(os.environ['REP_NUM'])
seed_value = SEED_VALUES[rep_num]
os.environ['PYTHONHASHSEED'] = str(seed_value)

#seed_value = int(os.environ['PYTHONHASHSEED'])
#print('Seed value: ', seed_value, type(os.environ['PYTHONHASHSEED']))
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

st.set_page_config(layout="wide")

app = MultiPage()
app.add_page("Home", home.app)
app.add_page("Image modification", image_modification.app)
app.add_page("LiDAR modification", lidar_modification.app)
app.add_page("Results", results.app)

app.run()