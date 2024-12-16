from huggingface_hub import upload_folder, create_repo

# Replace with your Hugging Face API token
hf_token = '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'


for layer in [3, 7, 11, 15]:
    for s in [100, 200, 400]:
        for a in [100, 300, 500, 1200]:
            layers = f'{layer-2},{layer-1},{layer}'
            saved_model_name = f'gemma_2_2b_it_s{s}_a{a}_layer{layer}'
                        
            # Replace with your model repo name and path to your model
            model_repo_name = f"yeutong/{saved_model_name}"
            # model_path = "/workspace/unlearning/outputs/checkpoints"
            model_path = f"~/unlearning/wmdp/models/{saved_model_name}"


            # Create the repository if it doesn't exist
            create_repo(repo_id=model_repo_name, private=True, token=hf_token, exist_ok=True)

            # Upload the folder
            upload_folder(
                folder_path=model_path,
                repo_id=model_repo_name,
                token=hf_token,
                repo_type='model',
                commit_message="Initial commit"
            )