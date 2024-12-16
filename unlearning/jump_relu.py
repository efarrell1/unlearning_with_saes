import torch.nn as nn
import numpy as np
from sae.config import Config
from huggingface_hub import hf_hub_download
import torch
from huggingface_hub import HfFileSystem


class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

    @torch.no_grad()
    def get_test_loss(self, batch_tokens, model):
        """
        A method for running the model with the SAE activations in order to return the loss.
        returns per token loss when activations are substituted in.
        """
        
        def standard_replacement_hook(activations, hook):
            activations = self.forward(activations).to(activations.dtype)
            return activations
        

        replacement_hook = standard_replacement_hook
        
        ce_loss_with_recons = model.run_with_hooks(
            batch_tokens,
            return_type="loss",
            fwd_hooks=[(self.cfg.hook_point, replacement_hook)],
        )
        
        ce_loss_wo_recons = model(batch_tokens, return_type="loss", prepend_bos=False)
        
        return ce_loss_with_recons, ce_loss_wo_recons


def load_gemma2_2b_sae(layer, l0=None, width=16):
    if l0 is None:
        l0, _ = get_gemma2_2b_SAE_path(layer, width=width)
        
    filename = "layer_" + str(layer) + "/width_" + str(width) + "k/average_l0_" + str(l0) + "/params.npz"
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res",
        filename=filename,
        local_dir='/workspace/weights/release-preview/gemmascope-2b-pt-res',
        force_download=False,
    )

    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)

    sae.cfg = Config()
    # sae.cfg.hook_name = "blocks." + str(layer) + ".hook_resid_post"
    sae.cfg.hook_point = "blocks." + str(layer) + ".hook_resid_post"
    sae.cfg.d_in = 2304

    return sae.to("cuda")

def get_gemma2_2b_SAE_path(layer, width=16, closest_l0=100):
    fs = HfFileSystem()
    all_paths = fs.glob("google/gemma-scope-2b-pt-res/**/params.npz")
    
    candidate_paths = [p for p in all_paths if f'layer_{layer}/width_{width}k/average_l0_' in p]
    
    # get the l0 value from the path
    l0_values = [int(p.split('average_l0_')[1].split('/')[0]) for p in candidate_paths]
    # print(l0_values)
    # print(*candidate_paths, sep='\n')
    
    # find the one closest to closest_l0
    idx = np.argmin(np.abs(np.array(l0_values) - closest_l0))
    desire_l0 = l0_values[idx]
    desire_path = candidate_paths[idx]
    
    print(f"Found SAE with l0={desire_l0} at path {desire_path}")
    
    return desire_l0, desire_path
