from tqdm import tqdm

import torch

from utils.shared_functions import init_dataset, InitNets
from utils.utils import set_seed

from options import SemanticSuPerOptions

options = SemanticSuPerOptions()
opt = options.parse()

torch.cuda.set_device(opt.gpu) # Set GPU.
set_seed(opt.seed)

def main():

    testloader = init_dataset(opt) # Init data loader.
    models = InitNets(opt) # Init models.

    for inputs in tqdm(testloader):
        models.super(models, inputs)
    
if __name__ == '__main__':
    main()