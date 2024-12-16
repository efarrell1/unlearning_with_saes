SAE_MAPPING = {
    "blocks.24.hook_resid_pre" : "desert-oath-8/sparse_autoencoder_EleutherAI/pythia-2.8b-deduped_blocks.24.hook_resid_pre_s20480_92155904.pt",
    "blocks.10.hook_resid_pre" : "absurd-dust-9/sparse_autoencoder_EleutherAI/pythia-2.8b-deduped_blocks.10.hook_resid_pre_s20480_92155904.pt",
    "blocks.11.hook_resid_pre" : "polished-wildflower-12/sparse_autoencoder_EleutherAI/pythia-2.8b-deduped_blocks.11.hook_resid_pre_s20480_327675904.pt",
    "blocks.11.hook_resid_pre_finetuned_hp" : "devout-surf-31/sparse_autoencoder_EleutherAI/pythia-2.8b-deduped_blocks.11.hook_resid_pre_s20480_40960000.pt",
    "blocks.0.hook_mlp_out_fulltrain_hp" : "avid-serenity-21/sparse_autoencoder/pythia-2.8b-deduped_blocks.0.hook_mlp_out_s5120_100M_tokens.pt",
    "blocks.5.hook_resid_pre_fulltrain_hp" : "classic-puddle-31/sparse_autoencoder/pythia-2.8b-deduped_blocks.0.hook_mlp_out_s5120_100M_tokens.pt",
    "blocks.5.hook_resid_pre_100M_pile" : "ancient-feather-33/sparse_autoencoder/pythia-2.8b-deduped_blocks.0.hook_mlp_out_s5120_100M_tokens_thepile_L0-23.pt",
    "blocks.5.hook_resid_pre" : "fragrant-disco-34/sparse_autoencoder/pythia-2.8b-deduped_blocks.0.hook_mlp_out_s20480_300M_tokens_thepile_L0-16.pt",
    "blocks.5.hook_resid_pre_finetuned_hp" : "worthy-morning-35/sparse_autoencoder/pythia-2.8b-deduped_blocks.0.hook_mlp_out_s20480_300M_tokens_thepile_70M_tokens_HP_L0-9.pt",
    "llama-2b_blocks.16.hook_resid_pre" : "lunar-fire-36_llama/sparse_autoencoder/Llama-2-7b-hf_blocks.16.hook_mlp_out_s32768_200M_tokens_openwebtext.pt",
    "gemma_2b_it_resid_pre_9": "jolly-dream-40/sparse_autoencoder_gemma-2b-it_blocks.9.hook_resid_pre_s16384_127995904.pt",
    "gemma_2_2b_it_resid_pre_9": "devoted-flower-44/sparse_autoencoder_gemma-2-2b-it_blocks.12.hook_resid_pre_s18432_481275904.pt",

}
REPO_ID = "eoinf/unlearning_saes"

GEMMA_INST_FORMAT = (
    "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
)

MIXTRAL_INST_FORMAT = (
    " [INST] {prompt} [/INST]"
)


# def convert_wmdp_data_to_prompt(datapoint, prompt_format=None, few_shot=False, few_shot_datapoint=None):
PRE_WMDP_BIO = "The following are multiple choice questions (with answers) about biology.\n"
PRE_WMDP_CYBER = "The following are multiple choice questions (with answers) about cyber security.\n"
PRE_QUESTION_FORMAT = "The following are multiple choice questions (with answers) about {subject}.\n"

gemma_2b_it_rmu_model_names = ['eoinf/gemma_2b_it_rmu_6',
                               'eoinf/gemma_2b_it_rmu_10',
                               'eoinf/gemma_2b_it_rmu_30',
                               'eoinf/gemma_2b_it_rmu_60',
                               'eoinf/gemma_2b_it_rmu_100']

MCQ_BATCH_SIZE = 6
