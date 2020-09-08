LBL_COLUMN = ['lbl']
CAT_COLUMNS = [f'cat{i}' for i in range(26)]
NUM_COLUMNS = [f'num{i}' for i in range(13)]
COLUMNS = LBL_COLUMN + NUM_COLUMNS + CAT_COLUMNS
FEATURE_COLUMNS = NUM_COLUMNS + CAT_COLUMNS
COLUMN_DEFAULTS = [0] * 14 + ['thisisdefault'] * 26
