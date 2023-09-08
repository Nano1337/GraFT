# GraFT: Gradual Fusion Transformer for Multimodal Re-Identification


arXiv (coming soon)

Official PyTorch implementation and pre-trained models for GraFT: Gradual Fusion Transformer for Multimodal Re-Identification 

We introduce the Gradual Fusion Transformer (**GraFT**), a cutting-edge model tailored for Multimodal Object Re-Identification (ReID). Traditional ReID models exhibit scalability constraints when handling multiple modalities due to their heavy reliance on late fusion, delaying the merging of insights from various modalities. GraFT tackles this by utilizing learnable fusion tokens which guide self-attention across encoders, adeptly capturing nuances of both modality-specific and object-centric features. Complementing its core design, GraFT is bolstered with an innovative training paradigm and an augmented triplet loss, refining the feature embedding space for ReID tasks. Our extensive ablation studies empiricaly validate our architectural design choices, proving GraFT's consistent outperformance against prevailing multimodal ReID benchmarks. 

## Datasets and Results

We used the [RGBNT100](https://drive.google.com/file/d/1ssrNqRNiOi2XHqt6JPsjptXWDJuFba9A/view?usp=sharing) and [RGBN300](https://drive.google.com/file/d/11QUGw_cwrEAa9chqxJc1WB3C4c0bgd4E/view?usp=sharing) datasets to benchmark against other algorithms. You may see our results in the following table: 


<table>
  <thead>
    <tr>
      <th>Method</th>
      <th colspan="5" style="text-align:center;">RGBNT100</th>
      <th colspan="5" style="text-align:center;">RGBN300</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td>
      <td align="center">Params</td>
      <td align="center">mAP</td>
      <td align="center">R1</td>
      <td align="center">R5</td>
      <td align="center">R10</td>
      <td align="center">Params</td>
      <td align="center">mAP</td>
      <td align="center">R1</td>
      <td align="center">R5</td>
      <td align="center">R10</td>
    </tr>
    <tr>
      <td>HAMNet</td>
      <td align="center">78M</td>
      <td align="center">65.4</td>
      <td align="center">85.5</td>
      <td align="center">87.9</td>
      <td align="center">88.8</td>
      <td align="center">52M</td>
      <td align="center">61.9</td>
      <td align="center">84.0</td>
      <td align="center">86.0</td>
      <td align="center">87.0</td>
    </tr>
    <tr>
      <td>DANet</td>
      <td align="center">78M</td>
      <td align="center">N/A</td>
      <td align="center">N/A</td>
      <td align="center">N/A</td>
      <td align="center">N/A</td>
      <td align="center">52M</td>
      <td align="center">71.0</td>
      <td align="center">89.9</td>
      <td align="center">90.9</td>
      <td align="center">91.5</td>
    </tr>
    <tr>
      <td>GAFNet</td>
      <td align="center">130M</td>
      <td align="center">74.4</td>
      <td align="center">93.4</td>
      <td align="center">94.5</td>
      <td align="center">95.0</td>
      <td align="center">130M</td>
      <td align="center">72.7</td>
      <td align="center">91.9</td>
      <td align="center">93.6</td>
      <td align="center">94.2</td>
    </tr>
    <tr>
      <td>Multi-Stream ViT</td>
      <td align="center">274M</td>
      <td align="center">74.6</td>
      <td align="center">91.3</td>
      <td align="center">92.8</td>
      <td align="center">93.5</td>
      <td align="center">187M</td>
      <td align="center">73.7</td>
      <td align="center">91.9</td>
      <td align="center">94.1</td>
      <td align="center">94.8</td>
    </tr>
    <tr>
      <td><b>GraFT (Ours)</b></td>
      <td align="center">101M</td>
      <td align="center"><b>76.6</b></td>
      <td align="center"><b>94.3</b></td>
      <td align="center"><b>95.3</b></td>
      <td align="center"><b>96.0</b></td>
      <td align="center">97M</td>
      <td align="center"><b>75.1</b></td>
      <td align="center"><b>92.1</b></td>
      <td align="center"><b>94.5</b></td>
      <td align="center"><b>95.2</b></td>
    </tr>
  </tbody>
</table>


## Catalog

- [ ] Release Pre-trained models 


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
