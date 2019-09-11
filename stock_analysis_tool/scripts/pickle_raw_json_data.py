from models import DataPreprocessor

DataPreprocessor()\
    .load_from_raw_json()\
    .expand()\
    .extract_sequence(year_cutoff=15)\
    .split()\
    .get(save=True)
