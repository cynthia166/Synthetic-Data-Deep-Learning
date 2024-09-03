import pickle
arf_root = "/home-local2/cyyba.extra.nobkp/Synthetic-Data-Deep-Learning/generative_input/FORED_fixedr.pkl"

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


arf = load_pickle(arf_root)

