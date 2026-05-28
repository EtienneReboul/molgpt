import argparse
import moses
import pandas as pd
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required = True, help="name of the generated dataset")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = pd.read_csv(args.path)

    test = moses.get_all_metrics(list(data['smiles'].values), device = device)

    print(args.path)
    print(test)
    print('*'*50)
