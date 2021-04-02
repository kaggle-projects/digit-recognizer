from utensil import get_logger
import pandas as pd
from sklearn import svm

logger = get_logger(__name__)


def get_xy(tr_path, te_path):
    x = pd.read_csv(tr_path).append(pd.read_csv(te_path))
    y = x['label']
    x = x.drop(columns='label')


