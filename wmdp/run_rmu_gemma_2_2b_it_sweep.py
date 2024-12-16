# %%
import subprocess

for layer in [3, 7, 11, 15]:
    for s in [100, 200, 400]:
        for a in [100, 300, 500, 1200]:
            layers = f'{layer-2},{layer-1},{layer}'
            saved_model_name = f'gemma_2_2b_it_s{s}_a{a}_layer{layer}'

            # Running the unlearning script
            unlearn_command = [
                "python3", "-m", "rmu.unlearn",
                "--model_name", "google/gemma-2-2b-it",
                "--max_num_batches", "300",
                "--batch_size", "3",
                "--retain_corpora", "wikitext",
                "--forget_corpora", "bio-forget-corpus",
                "--steering_coeffs", str(s),
                "--alpha", str(a),
                "--layer_id", str(layer),
                "--layer_ids", layers,
                "--lr", "5e-5",
                "--seed", "42",
                "--output_dir", f"models/{saved_model_name}"
            ]
            subprocess.run(unlearn_command)

            # Running the evaluation script
            eval_command = [
                "lm-eval", "--model", "hf",
                "--model_args", f"pretrained=models/{saved_model_name}",
                "--tasks", "wmdp_bio,mmlu_college_biology",
                "--batch_size", "8",
                "--output_path", "eval_results"
            ]
            subprocess.run(eval_command)
