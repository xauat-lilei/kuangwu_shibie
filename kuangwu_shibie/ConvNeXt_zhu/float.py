from  ConvNeXt_model import convnext_tiny

import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
    net =convnext_tiny(num_classes=26)
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
