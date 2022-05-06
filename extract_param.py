import tensorflow as tf
import os
import pickle
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config


def parse_args():
    parser = argparase.ArgumentParser()
    parser.add_argument('pkl_model_path', type=str, help='Path to the pre-trained pkl-model.')
    parser.add_argument('output_name', type=str, help='Name of the dict of params.')
    return parser.parse_args()

def main():
    args = parse_args()

    tflib.init_tf()
    with open(args.pkl_model_path, 'rb') as f:
        _, _, _D, _ = pickle.load(f)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    param_name = []
    for key, value in _D.vars.items():
      param_name.append(key)

    d_param = {}
    for i in range(len(param_name)):
      if ('weight' in param_name[i]):
        print(param_name[i])
        if 'Dense' in param_name[i]:
          param = _D.get_var(param_name[i])
          param = param.transpose(1, 0)
          print(param.shape)
          d_param[param_name[i]] = param
        else:
          param = _D.get_var(param_name[i])
          param = param.transpose(3, 2, 0, 1)
          print(param.shape)
          d_param[param_name[i]] = param
      else:
        param = _D.get_var(param_name[i])
        d_param[param_name[i]] = param

    with open(args.output_model_name, 'wb') as f:
        pickle.dump(d_param, f)

if __name__ == '__main__':
    main()
