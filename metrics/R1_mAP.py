from typing import Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE
import plotly
import plotly.express as px
import torch
import torch.distributed as dist


def eval_func(distmat: np.array,
              q_pids: List[int],
              g_pids: List[int],
              q_camids: List[int],
              g_camids: List[int],
              max_rank: int = 50) -> Tuple[np.array, float]:
    """Evaluate the Cumulative Matching Characteristics (CMC) and the mean Average Precision (mAP).

    Args:
        distmat: The distance matrix between each query and gallery sample.
        q_pids: The person identifiers for each query sample.
        g_pids: The person identifiers for each gallery sample.
        q_camids: The camera identifiers for each query sample.
        g_camids: The camera identifiers for each gallery sample.
        max_rank: The maximum rank to be considered for CMC calculation.

    Returns:
        all_cmc: The CMC for each query.
        mAP: The mean Average Precision.
    """
    num_q, num_g = distmat.shape
    assert num_q > 0, "Error: all query identities do not appear in gallery"

    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum() / (np.arange(len(orig_cmc)) + 1.)
        tmp_cmc *= orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP:
    """Computes rank-1 matching rate (R1) and mean Average Precision (mAP) for given features, person and camera id.

    Args:
        fabric: Computational device to use for calculations.
        num_query: Number of queries.
        max_rank: Maximum rank to consider for CMC calculation.
        feat_norm: Whether to normalize the features.
    """

    def __init__(self,
                 fabric: Any,
                 cfgs: dict,
                 num_query: int,
                 max_rank: int = 50,
                 feat_norm: str = 'yes',
                 process_group: Optional[dist.ProcessGroup] = None):
        self.fabric = fabric
        self.cfgs = cfgs
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.process_group = process_group
        self.reset()

    def gather_tensors(self, tensor):
        gathered = [torch.zeros_like(tensor) for _ in range(self.fabric.world_size)]
        dist.all_gather(gathered, tensor, group=self.process_group)
        gathered = torch.cat(gathered, dim=0)
        return gathered

    def gather_lists(self, data):
        data = torch.tensor(data).to(self.fabric.device)
        gathered = [torch.zeros_like(data) for _ in range(self.fabric.world_size)]
        dist.all_gather(gathered, data, group=self.process_group)
        gathered = torch.cat(gathered, dim=0).tolist()
        return gathered

    def reset(self):
        """Resets the features, person identifiers, and camera identifiers."""
        self.feats = self.fabric.to_device(torch.tensor([]))
        self.pids = []
        self.camids = []

    def update(self, feat: torch.Tensor, pid: List[int], camid: List[int]):
        """Updates the features, person identifiers, and camera identifiers.

        Args:
            feat: New features.
            pid: New person identifiers.
            camid: New camera identifiers.
        """
        if len(self.cfgs.gpus) > 1: 
            feat = self.gather_tensors(feat)
            pid = self.gather_lists(pid)
            camid = self.gather_lists(camid)

        if self.fabric.is_global_zero:
            self.feats = torch.cat([self.feats, feat], dim=0)
            self.pids.extend(pid)
            self.camids.extend(camid)

    def compute_umap_plotly(self,
                            save_path: str = "./visualization/",
                            dims: int = 2,
                            reduction_method: str = 'tsne'):
        """Visualizes features in 2D or 3D using UMAP or t-SNE and plots using Plotly.

        Args:
            save_path: Path where to save the visualizations.
            dims: Number of dimensions for the visualization (either 2 or 3).
            reduction_method: 'umap' or 'tsne' for the dimensionality reduction method.
        """

        def create_reducer(n_components: int):
            if reduction_method == 'tsne':
                return TSNE(n_components=n_components, random_state=42)
            else:
                return umap.UMAP(n_components=n_components, random_state=42)

        def create_dataframe(embeddings: np.array):
            return pd.DataFrame({
                'x': embeddings[:, 0],
                'y': embeddings[:, 1],
                'z': embeddings[:, 2] if dims == 3 else None,
                'pid': self.pids,
                'camid': self.camids
            })

        def create_plot_2d(df: pd.DataFrame):
            fig = plotly.graph_objects.Figure(
                data=plotly.graph_objects.Scatter(x=df['x'],
                                                  y=df['y'],
                                                  mode='markers',
                                                  marker=dict(
                                                      size=6,
                                                      color=df['pid'],
                                                      colorscale='Rainbow',
                                                      showscale=True),
                                                  text=df['pid']))
            fig.update_layout(
                title=f"{reduction_method.upper()} projection of the features",
                xaxis_title=f'{reduction_method.upper()}1',
                yaxis_title=f'{reduction_method.upper()}2',
                autosize=False,
                width=1000,
                height=1000,
                margin=dict(l=65, r=50, b=65, t=90))
            fig.write_image(save_path + f'{reduction_method}_2d.png')

        def create_plot_3d(df: pd.DataFrame):
            fig = plotly.graph_objects.Figure(
                data=plotly.graph_objects.Scatter3d(x=df['x'],
                                                    y=df['y'],
                                                    z=df['z'],
                                                    mode='markers',
                                                    marker=dict(
                                                        size=6,
                                                        color=df['pid'],
                                                        colorscale='Rainbow',
                                                        showscale=True),
                                                    text=df['pid']))
            fig.update_layout(
                title=f"{reduction_method.upper()} projection of the features",
                scene=dict(xaxis_title=f'{reduction_method.upper()}1',
                           yaxis_title=f'{reduction_method.upper()}2',
                           zaxis_title=f'{reduction_method.upper()}3'),
                autosize=False,
                width=1000,
                height=1000,
                scene_aspectmode='cube',
                margin=dict(l=65, r=50, b=65, t=90))
            plotly.offline.plot(fig,
                                filename=save_path + f'{reduction_method}_3d.html')

        if self.feats is None or self.pids is None:
            raise ValueError(
                "self.feats or self.pids is None or not in the expected format."
            )

        reducer = create_reducer(dims)
        embeddings = reducer.fit_transform(self.feats.cpu().numpy())
        df = create_dataframe(embeddings)
        df['pid'] = df['pid'].astype('category')
        palette = px.colors.qualitative.Plotly
        df['color'] = df['pid'].cat.codes.map(
            lambda x: palette[x % len(palette)])

        if dims == 2:
            create_plot_2d(df)
            df.to_csv(save_path + f'{reduction_method}_2d.csv', index=False)
        elif dims == 3:
            create_plot_3d(df)
            df.to_csv(save_path + f'{reduction_method}_3d.csv', index=False)
        else:
            raise ValueError("dims must be 2 or 3")

    def compute(self) -> Tuple[np.array, float]:
        """Computes the CMC and mAP values for the updated features, person identifiers, and camera identifiers.
            Only compute on global rank zero. 

        Returns:
            cmc: The CMC for each query.
            mAP: The mean Average Precision.
        """
        if self.fabric.is_global_zero:
            if self.feat_norm == 'yes':
                feats = torch.nn.functional.normalize(self.feats, dim=1, p=2)
            qf = feats[:self.num_query]
            q_pids = np.asarray(self.pids[:self.num_query])
            q_camids = np.asarray(self.camids[:self.num_query])
            gf = feats[self.num_query:]
            g_pids = np.asarray(self.pids[self.num_query:])
            g_camids = np.asarray(self.camids[self.num_query:])

            m, n = qf.shape[0], gf.shape[0]
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
            distmat = distmat.cpu().numpy()

            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
            return cmc, mAP
        else:
            return 0, 0
