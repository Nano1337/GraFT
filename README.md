# GraFT: Gradual Fusion Transformer for Multimodal Re-Identification


['arXiv (coming soon)'](https://arxiv.org)

Official PyTorch implementation and pre-trained models for GraFT: Gradual Fusion Transformer for Multimodal Re-Identification 

We introduce the Gradual Fusion Transformer (**GraFT**), a cutting-edge model tailored for Multimodal Object Re-Identification (ReID). Traditional ReID models exhibit scalability constraints when handling multiple modalities due to their heavy reliance on late fusion, delaying the merging of insights from various modalities. GraFT tackles this by utilizing learnable fusion tokens which guide self-attention across encoders, adeptly capturing nuances of both modality-specific and object-centric features. Complementing its core design, GraFT is bolstered with an innovative training paradigm and an augmented triplet loss, refining the feature embedding space for ReID tasks. Our extensive ablation studies empiricaly validate our architectural design choices, proving GraFT's consistent outperformance against prevailing multimodal ReID benchmarks. 

## Datasets and Results

We used the [RGBNT100](https://drive.google.com/file/d/1ssrNqRNiOi2XHqt6JPsjptXWDJuFba9A/view?usp=sharing) and [RGBN300](https://drive.google.com/file/d/11QUGw_cwrEAa9chqxJc1WB3C4c0bgd4E/view?usp=sharing) datasets to benchmark against other algorithms. You may see our results in the following table: 


<table>
<thead>
  <tr>
    <th>Method</th>
    <th>Classif. (@1)</th>
    <th colspan="7">Semantic Segmentation (mIoU)</th>
    <th>Depth (δ1)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td align="center"> ImageNet-1K<br>(RGB)<br></td>
    <td align="center">ADE20K<br>(RGB)<br></td>
    <td align="center" colspan="3">Hypersim<br>(RGB / D / RGB + D)<br></td>
    <td align="center"colspan="3">NYUv2<br>(RGB / D / RGB + D)<br></td>
    <td align="center">NYUv2<br>(RGB)<br></td>
  </tr>
  <tr>
    <td>Sup. (DeiT)</td>
    <td align="center">81.8</td>
    <td align="center">45.8</td>
    <td align="center">33.9</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">50.1</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">80.7</td>
  </tr>
  <tr>
    <td>MAE</td>
    <td align="center"><b>83.3</b></td>
    <td align="center"><b>46.2</b></td>
    <td align="center">36.5</td>
    <td align="center">-</td>
    <td align="center">-<br></td>
    <td align="center">50.8</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">85.1</td>
  </tr>
  <tr>
    <td><b>MultiMAE</b></td>
    <td align="center"><b>83.3</b></td>
    <td align="center"><b>46.2</b></td>
    <td align="center"><b>37.0</b></td>
    <td align="center"><b>38.5</b></td>
    <td align="center"><b>47.6</b></td>
    <td align="center"><b>52.0</b></td>
    <td align="center"><b>41.4</b></td>
    <td align="center"><b>56.0</b></td>
    <td align="center"><b>86.4</b></td>
  </tr>
</tbody>
</table>


## Catalog

- [] Release Pre-trained models 


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
