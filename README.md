# GraFT: Gradual Fusion Transformer for Multimodal Re-Identification

[Haoli Yin](https://haoliyin.me), [Emily Li](https://emilyjiayaoli.me), [Eva Schiller](https://www.linkedin.com/in/eva-schiller/), [Luke McDermott](https://scholar.google.com/citations?user=l_z4cj0AAAAJ&hl=en), [Daniel Cummings](https://scholar.google.com.au/citations?user=Dud0vLwAAAAJ&hl=en)



## Setup 


1. Clone from the correct branch
    ```bash
    git clone -b cleaned-final ssh://git@192.168.100.16:222/ModernIntelligence/research-GraFT.git
    ```

2. Create python venv
    ```bash
    python -m venv <your-name>
    ```

3. Activate venv
    ```bash
    source <your-name>/bin/activate
    ```

4. Install requirements
    ```bash
    pip install -r requirements.txt
    ```

5. Install hugginface transformers from source to use DeiT
    ```bash
    pip install git+https://github.com/huggingface/transformers
    ```

6. Check train_optuna.sh to run with correct configs/yaml file
    ```bash
    sh ./train_optuna.sh
    ```


## Pre-Experiment Checklist

- Check configs

```yaml
use_optuna: True

gpus: [0, 1, 2, 3]

model_name: "transformer_baseline_reid_v2”

# Weights and Biases Logging
use_wandb: True
wandb_project: "mm-mafia-reid-baseline" # generally stays same
study_name: "transformer_baseline_v2_rn100_param=5m_no_downsampling_patch=24" # experiment level
wandb_run_name: "transformer_baseline_v2_rn100_param=5m_no_downsampling_patch=24" # keep same as study_name
wandb_trial_name: "elarger patch size=64, lower seq_len=4, e-5-6lr, transformer_encoder=3" # trial_name under study
```

- If using optuna for hyperparameter search:
    - use_optuna, if True then make sure to use train_optuna.py
        - check the lr and weight decay range specified at the beginning of train_optuna.py
- Activate virtual environment
    
    ```bash
    source <your-name>/bin/activate
    ```
    
- Run job scheduler interface (optional)
    ```bash
    python webapp_ui/app.py
    ```
