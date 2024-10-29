import os, sys

import torch

sys.path.append(os.getcwd())

def save_box_quats(box_quats, id, output_path):
    torch.save(box_quats, os.path.join(output_path, "box_quats/{}.pt".format(id)))
