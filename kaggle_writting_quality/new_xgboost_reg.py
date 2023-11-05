from sklearn.preprocessing import 
from sklearn.feature_selection import SelectPercentile

from data_funs import clean_log_data

class CONFIG:
    DATA_DIR_PATH = r"D:\dev\linking-writing-processes-to-writing-quality"
    COLS_TO_DROP = ["id", "score"]
    RANDOM_SEED = 2023
    TEST_SIZE = 0.2
    ESTIMATORS = [("clf", XGBRegressor(random_state=RANDOM_SEED))]
    CV = 4
    N_ITER = 5000
    PAUSE_THRESHOLD = 2000
    

if __name__ == "__main__":
    