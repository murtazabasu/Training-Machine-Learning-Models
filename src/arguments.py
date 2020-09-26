import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--create_folds", default=False)
    parser.add_argument("--n_splits", type=int, help="input the number \
                        of splits for creating folds")
 
    args = parser.parse_args()

    return args