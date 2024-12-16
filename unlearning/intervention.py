import torch
import einops

from torch import Tensor
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
from sae.sparse_autoencoder import SparseAutoencoder
from contextlib import contextmanager
from functools import partial
# from sae_lens import SAE
# from sae.sparse_autoencoder import SparseAutoencoder as SAE
from unlearning.jump_relu import JumpReLUSAE

import numpy as np

def remove_resid_SAE_features(
    resid: Float[Tensor, "batch seq d_model"], 
    hook: HookPoint,
    sae: SparseAutoencoder | JumpReLUSAE,
    features_to_ablate: list[int],
    multiplier: float = 1.0
):
    """
    Given a list of feature indices, this hook function :
        1. Projects the residual stream activations onto the feature directions
        2. Subtracts the projected feature activations (weighted by a multiplier) from the activations.
    """

    if len(features_to_ablate) > 0:
        #if this feature is aligned with the bias, we don't want to get extra activation from the bias.
        # This probably won't matter for almost any high-dimensional vector, but it might for a few.
        resid_copy = resid.clone() - sae.b_dec

        # Old code
        #get normalized decoder vectors
        # decoder_vecs = sae.W_dec[features_to_ablate,:]
        # norms = torch.norm(decoder_vecs, dim=-1, keepdim=True)
        # decoder_vecs /= norms

        # # dot product scalar * unit feature vector = projection vector
        # dots = einops.einsum(resid_copy, decoder_vecs, "batch seq d_model, feats d_model -> batch seq feats")
        # dots *= multiplier #allow us to crank up or down how much we remove
        # projections = einops.einsum(dots, decoder_vecs, "batch seq feats, feats d_model -> batch seq d_model")

        # resid = resid - projections
        
        # Do the features one at a time
        for j in range(1):
            for feature in features_to_ablate:
                decoder_vec = sae.W_dec[feature,:]
                norm = torch.norm(decoder_vec, dim=-1, keepdim=True)
                decoder_vec /= norm
    
                # dot product scalar * unit feature vector = projection vector
                dot = einops.einsum(resid_copy, decoder_vec, "batch seq d_model, d_model -> batch seq")
                projection = einops.einsum(dot, decoder_vec, "batch seq, d_model -> batch seq d_model")
    
                #remove projection without scaling from resid_copy to avoid feature interference.
                resid_copy = resid_copy - projection
    
                #remove projection from resid with appropriate scaling
                resid = resid - multiplier*projection
    return resid


def anthropic_remove_resid_SAE_features(
    resid: Float[Tensor, "batch seq d_model"], 
    hook: HookPoint,
    sae: SparseAutoencoder | JumpReLUSAE,
    features_to_ablate: list[int],
    multiplier: float = 1.0,
):
    """
    Given a list of feature indices, this hook function removes feature activations in a manner similar to the one
    used in "Scaling Monosemanticity": https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#appendix-methods-steering
    """

    if len(features_to_ablate) > 0:

        with torch.no_grad():
            #adjust feature activations with scaling (multiplier = 0 just ablates the feature)
            if isinstance(sae, SparseAutoencoder):
                feature_activations, reconstruction = sae(resid)
            elif isinstance(sae, JumpReLUSAE):
                reconstruction = sae(resid)
                feature_activations = sae.encode(resid)
            else:
                raise ValueError("sae must be an instance of SparseAutoencoder or SAE")

            error = resid - reconstruction

            feature_activations[:, :, features_to_ablate] -= multiplier * feature_activations[:, :, features_to_ablate]
            #finish modified forward pass for feature clamping

            modified_reconstruction = einops.einsum(feature_activations, sae.W_dec, "... d_sae, d_sae d_in -> ... d_in")\
                + sae.b_dec
            
            # Unscale outputs if needed:
            # if sae.input_scaling_factor is not None:
            #     modified_reconstruction = modified_reconstruction / sae.input_scaling_factor
            resid = error + modified_reconstruction
        return resid


def anthropic_clamp_resid_SAE_features(
    resid: Float[Tensor, "batch seq d_model"], 
    hook: HookPoint,
    sae: SparseAutoencoder | JumpReLUSAE,
    features_to_ablate: list[int],
    multiplier: float = 1.0,
    random: bool = False,
):
    """
    Given a list of feature indices, this hook function removes feature activations in a manner similar to the one
    used in "Scaling Monosemanticity": https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#appendix-methods-steering
    This version clamps the feature activation to the value(s) specified in multiplier
    """

    if len(features_to_ablate) > 0:

        with torch.no_grad():
            #adjust feature activations with scaling (multiplier = 0 just ablates the feature)
            if isinstance(sae, SparseAutoencoder):
                feature_activations, reconstruction = sae(resid)
            elif isinstance(sae, JumpReLUSAE):
                reconstruction = sae(resid)
                feature_activations = sae.encode(resid)
            else:
                try:
                    import sys
                    sys.path.append('/root')

                    from dictionary_learning import AutoEncoder
                    from dictionary_learning.trainers.top_k import AutoEncoderTopK

                    if isinstance(sae, (AutoEncoder, AutoEncoderTopK)):
                        reconstruction = sae(resid)
                        feature_activations = sae.encode(resid)
                except:
                    raise ValueError("sae must be an instance of SparseAutoencoder or SAE")

            error = resid - reconstruction

            non_zero_features = feature_activations[:, :, features_to_ablate] > 0
            
            
            if not random:
                    
                if isinstance(multiplier, float) or isinstance(multiplier, int):
                    feature_activations[:, :, features_to_ablate] = torch.where(non_zero_features, -multiplier, feature_activations[:, :, features_to_ablate])
                else:
                    feature_activations[:, :, features_to_ablate] = torch.where(non_zero_features,
                                                                                -multiplier.unsqueeze(dim=0).unsqueeze(dim=0),
                                                                                feature_activations[:, :, features_to_ablate])    
                
            # set the next feature id's activations to the multiplier only if the previous feature id's
            # activations are positive
            else:
                assert isinstance(multiplier, float) or isinstance(multiplier, int)
                
                next_features_to_ablate = [(f + 1) % feature_activations.shape[-1] for f in features_to_ablate]
                feature_activations[:, :, next_features_to_ablate] = torch.where(
                    feature_activations[:, :, features_to_ablate] > 0,
                    -multiplier,
                    feature_activations[:, :, next_features_to_ablate]
                )   
                
            try:
                modified_reconstruction = einops.einsum(feature_activations, sae.W_dec, "... d_sae, d_sae d_in -> ... d_in")\
                    + sae.b_dec
            except:
                # SAEBench doesn't have W_dec and b_dec
                modified_reconstruction = sae.decode(feature_activations)
            
            # Unscale outputs if needed:
            # if sae.input_scaling_factor is not None:
            #     modified_reconstruction = modified_reconstruction / sae.input_scaling_factor
            resid = error + modified_reconstruction
        return resid


def anthropic_clamp_jump_relu_resid_SAE_features(
    resid: Float[Tensor, "batch seq d_model"], 
    hook: HookPoint,
    sae: SparseAutoencoder | JumpReLUSAE,
    features_to_ablate: list[int],
    multiplier: float = 1.0,
    jump: float = 0.0,
):
    """
    Given a list of feature indices, this hook function removes feature activations in a manner similar to the one
    used in "Scaling Monosemanticity": https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#appendix-methods-steering
    This version clamps the feature activation to the value(s) specified in multiplier and adds a jump Relu
    """

    if len(features_to_ablate) > 0:

        with torch.no_grad():
            #adjust feature activations with scaling (multiplier = 0 just ablates the feature)
            if isinstance(sae, SparseAutoencoder):
                feature_activations, reconstruction = sae(resid)
            elif isinstance(sae, SAE):
                reconstruction = sae(resid)
                feature_activations = sae.encode(resid)
            else:
                raise ValueError("sae must be an instance of SparseAutoencoder or SAE")

            feature_activations[feature_activations < jump] = 0

            error = resid - reconstruction

            non_zero_features = feature_activations[:, :, features_to_ablate] > 0
            
            if isinstance(multiplier, float) or isinstance(multiplier, int):
                feature_activations[:, :, features_to_ablate] = torch.where(non_zero_features, -multiplier, feature_activations[:, :, features_to_ablate])
            else:
                feature_activations[:, :, features_to_ablate] = torch.where(non_zero_features,
                                                                            -multiplier.unsqueeze(dim=0).unsqueeze(dim=0),
                                                                            feature_activations[:, :, features_to_ablate])

            modified_reconstruction = einops.einsum(feature_activations, sae.W_dec, "... d_sae, d_sae d_in -> ... d_in")\
                + sae.b_dec
            
            # Unscale outputs if needed:
            # if sae.input_scaling_factor is not None:
            #     modified_reconstruction = modified_reconstruction / sae.input_scaling_factor
            resid = error + modified_reconstruction
        return resid


def zero_ablate(
    resid: Float[Tensor, "batch seq d_model"], 
    hook: HookPoint,
    sae: SparseAutoencoder | JumpReLUSAE,
    features_to_ablate: list[int],
    multiplier: float = 0
):
    """
    Given a list of feature indices, this hook function removes feature activations in a manner similar to the one
    used in "Scaling Monosemanticity": https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#appendix-methods-steering
    """

    if len(features_to_ablate) > 0:

        with torch.no_grad():
            #adjust feature activations with scaling (multiplier = 0 just ablates the feature)
            if isinstance(sae, SparseAutoencoder):
                feature_activations, reconstruction = sae(resid)
            elif isinstance(sae, SAE):
                reconstruction = sae(resid)
                feature_activations = sae.encode(resid)
            else:
                raise ValueError("sae must be an instance of SparseAutoencoder or SAE")

            error = resid - reconstruction

            feature_activations[:, :, features_to_ablate] = 0
            #finish modified forward pass for feature clamping

            modified_reconstruction = einops.einsum(feature_activations, sae.W_dec, "... d_sae, d_sae d_in -> ... d_in")\
                + sae.b_dec
            
            # Unscale outputs if needed:
            # if sae.input_scaling_factor is not None:
            #     modified_reconstruction = modified_reconstruction / sae.input_scaling_factor
            resid = error + modified_reconstruction
        return resid
    
    
def mean_ablate(
    resid: Float[Tensor, "batch seq d_model"], 
    hook: HookPoint,
    sae: SparseAutoencoder | JumpReLUSAE,
    features_to_ablate: list[int],
    multiplier: float = 0
):
    """
    Given a list of feature indices, this hook function removes feature activations in a manner similar to the one
    used in "Scaling Monosemanticity": https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#appendix-methods-steering
    """

    if len(features_to_ablate) > 0:

        with torch.no_grad():
            #adjust feature activations with scaling (multiplier = 0 just ablates the feature)
            if isinstance(sae, SparseAutoencoder):
                feature_activations, reconstruction = sae(resid)
            elif isinstance(sae, SAE):
                reconstruction = sae(resid)
                feature_activations = sae.encode(resid)
            else:
                raise ValueError("sae must be an instance of SparseAutoencoder or SAE")

            error = resid - reconstruction

            feature_activations[:, :, features_to_ablate] = torch.mean(feature_activations[:, :, features_to_ablate], dim=(0, 1), keepdim=True)
            #finish modified forward pass for feature clamping

            modified_reconstruction = einops.einsum(feature_activations, sae.W_dec, "... d_sae, d_sae d_in -> ... d_in")\
                + sae.b_dec
            
            # Unscale outputs if needed:
            # if sae.input_scaling_factor is not None:
            #     modified_reconstruction = modified_reconstruction / sae.input_scaling_factor
            resid = error + modified_reconstruction
        return resid




def scale_feature_hook_hf(mod, inputs, outputs, sae, features_to_ablate, multiplier, random=False):
    resid = outputs[0].to(torch.float)
    reconstruction = sae(resid)
    feature_activations = sae.encode(resid)
    error = resid - reconstruction
    
    if random:
        raise NotImplementedError("Random scaling not implemented, use clamping!")
    
    feature_activations[:, :, features_to_ablate] -= multiplier * feature_activations[:, :, features_to_ablate]
    modified_reconstruction = feature_activations @ sae.W_dec + sae.b_dec
    
    resid = error + modified_reconstruction
    return (resid.to(torch.bfloat16), None)

def clamping_feature_hook_hf(mod, inputs, outputs, sae, features_to_ablate, multiplier, random=False):
    resid = outputs[0].to(torch.float)
    reconstruction = sae(resid)
    feature_activations = sae.encode(resid)
    error = resid - reconstruction
    
    # set the feature activations to the multiplier only if they are positive
    if not random:
        feature_activations[:, :, features_to_ablate] = torch.where(
            feature_activations[:, :, features_to_ablate] > 0,
            -multiplier,
            feature_activations[:, :, features_to_ablate]
        )
    # set the next feature id's activations to the multiplier only if the previous feature id's
    # activations are positive
    else:
        next_features_to_ablate = [(f + 1) % feature_activations.shape[-1] for f in features_to_ablate]
        feature_activations[:, :, next_features_to_ablate] = torch.where(
            feature_activations[:, :, features_to_ablate] > 0,
            -multiplier,
            feature_activations[:, :, next_features_to_ablate]
        )            
        

    modified_reconstruction = feature_activations @ sae.W_dec + sae.b_dec
    
    resid = error + modified_reconstruction
    return (resid.to(torch.bfloat16), None)


@contextmanager
def scaling_intervention(model, layer, sae, features_to_ablate, multiplier):
    """intervene on resid post at given layer"""
    handle = model.model.layers[layer].register_forward_hook(
        partial(scale_feature_hook_hf, sae=sae, features_to_ablate=features_to_ablate, multiplier=multiplier)
    )
    try:
        yield
    finally:
        handle.remove()
        
        
@contextmanager
def intervention(model, layer, sae, features_to_ablate, multiplier, intervention_type, random=False):
    """intervene on resid post at given layer"""
    assert intervention_type in ['scale', 'clamp']
    hook_func_map = {
        'scale': scale_feature_hook_hf,
        'clamp': clamping_feature_hook_hf,
    }
    hook_func = hook_func_map[intervention_type]
    
    handle = model.model.layers[layer].register_forward_hook(
        partial(hook_func, sae=sae, features_to_ablate=features_to_ablate, multiplier=multiplier, random=random)
    )
    try:
        yield
    finally:
        handle.remove()