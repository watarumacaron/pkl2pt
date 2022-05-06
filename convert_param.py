import pickle
import torch
from StyleGANDiscriminator import Discriminator_pt as D

def parse_args():
    parser = argparase.ArgumentParser()
    parser.add_argument('pkl_params_path', type=str, help='Path to the pkl-param.')
    parser.add_argument('output_model_name', type=str, help='Name of the params of pytorch-model.')
    return parser.parse_args()

def main():
    args = parse_args()
    discriminator = D(0, 256, 3)

    with open(args.pkl, 'rb') as f:
        d_params = pickle.load(f)
    print("Loading completed.")

    param_name = []
    for key in d_params.keys():
        param_name.append(key)
    param_name = param_name[1:]

    torch_param = discriminator.state_dict()
    count = 0
    param_name_pt = []
    for key in torch_param.keys():
        if 'resample' not in key:
            param_name_pt.append(key)
            count += 1
    print('Extraction of names of params completed.')

    for i in range(len(param_name)):
        param = d_param[param_name[i]]
        torch_param[param_name_pt[i]].data = torch.tensor(param)

    discriminator.load_state_dict(torch_param)

    torch.save(discriminator.state_dict(), args.output_model_name)
    print('Saved completely.')

if __name__ == '__main__':
    main()
