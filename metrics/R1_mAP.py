import torch 
import numpy as np
import umap
import matplotlib.pyplot as plt
import plotly

import plotly.express as px
import plotly.graph_objects as go

from sklearn.manifold import TSNE

import pandas as pd


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """
    This function calculates the Cumulative Matching Characteristics (CMC) and the mean Average Precision (mAP)
    for the provided query and gallery samples.

    Args:
        distmat (np.array): A distance matrix containing the pairwise distance between each query and gallery sample.
        q_pids (list): The person identifiers for each query sample.
        g_pids (list): The person identifiers for each gallery sample.
        q_camids (list): The camera identifiers for each query sample.
        g_camids (list): The camera identifiers for each gallery sample.
        max_rank (int, optional): The maximum rank to be considered for CMC calculation. Defaults to 50.

    Raises:
        AssertionError: If the number of queries is zero.

    Returns:
        np.array, float: An array representing the CMC for each query and the mAP.
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
    # print(all_cmc, num_valid_q)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP:
    """
    This class is used to calculate the rank-1 matching rate (R1) and the mean Average Precision (mAP) 
    for given features, person and camera identifiers.

    Args:
        fabric (any): The computational device to use for calculations.
        num_query (int): The number of queries.
        max_rank (int, optional): The maximum rank to consider for CMC calculation. Defaults to 50.
        feat_norm (str, optional): Whether to normalize the features. Defaults to 'yes'.

    Attributes:
        feats (torch.Tensor): The features tensor.
        pids (list): The person identifiers.
        camids (list): The camera identifiers.
    """

    def __init__(self, fabric, num_query, max_rank=50, feat_norm='yes'):
        self.fabric = fabric
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reset()

    def reset(self):
        """
        Resets the features tensor, person identifiers, and camera identifiers to their initial state.
        """

        self.feats = self.fabric.to_device(torch.tensor([]))
        self.pids = []
        self.camids = []

    def update(self, feat, pid, camid):
        """
        Updates the features tensor, person identifiers, and camera identifiers with new values.

        Args:
            feat (torch.Tensor): The new features tensor.
            pid (list): The new person identifiers.
            camid (list): The new camera identifiers.
        """
        self.feats = torch.cat([self.feats, feat], dim=0) # same as using stack
        self.pids.extend(pid)
        self.camids.extend(camid)

    def compute_umap_plotly(self, save_path="./visualization/", dims=2, reduction_method='tsne'):
        """
        Visualizes features in 2D or 3D using UMAP or t-SNE and plots using Plotly.

        :param dims: number of dimensions for the visualization (either 2 or 3)
        :param reduction_method: 'umap' or 'tsne'
        """

        def create_reducer(n_components):
            if reduction_method == 'tsne':
                return TSNE(n_components=n_components, random_state=42)
            else:
                return umap.UMAP(n_components=n_components, random_state=42)

        def create_dataframe(embeddings):
            return pd.DataFrame({
                'x': embeddings[:, 0],
                'y': embeddings[:, 1],
                'z': embeddings[:, 2] if dims == 3 else None,
                'pid': self.pids,
                'camid': self.camids
            })

        def create_plot_2d(df):
            fig = plotly.graph_objects.Figure(data=plotly.graph_objects.Scatter(
                x = df['x'],
                y = df['y'],
                mode='markers',
                marker=dict(size=6, color=df['pid'], colorscale='Rainbow', showscale=True),
                text=df['pid']
            ))
            fig.update_layout(
                title=f"{reduction_method.upper()} projection of the features",
                xaxis_title=f'{reduction_method.upper()}1',
                yaxis_title=f'{reduction_method.upper()}2',
                autosize=False,
                width=1000,
                height=1000,
                margin=dict(l=65, r=50, b=65, t=90)
            )
            fig.write_image(save_path + f'{reduction_method}_2d.png')

        def create_plot_3d(df):
            fig = plotly.graph_objects.Figure(data=plotly.graph_objects.Scatter3d(
                x = df['x'],
                y = df['y'],
                z = df['z'],
                mode='markers',
                marker=dict(size=6, color=df['pid'], colorscale='Rainbow', showscale=True),
                text=df['pid']
            ))
            fig.update_layout(
                title=f"{reduction_method.upper()} projection of the features",
                scene=dict(xaxis_title=f'{reduction_method.upper()}1', 
                        yaxis_title=f'{reduction_method.upper()}2', 
                        zaxis_title=f'{reduction_method.upper()}3'),
                autosize=False,
                width=1000,
                height=1000,
                scene_aspectmode='cube',
                margin=dict(l=65, r=50, b=65, t=90)
            )
            plotly.offline.plot(fig, filename=save_path + f'{reduction_method}_3d.html')

        if self.feats is None or self.pids is None:
            raise ValueError("self.feats or self.pids is None or not in the expected format.")

        reducer = create_reducer(dims)
        embeddings = reducer.fit_transform(self.feats.cpu().numpy())
        df = create_dataframe(embeddings)
        df['pid'] = df['pid'].astype('category')
        palette = px.colors.qualitative.Plotly
        df['color'] = df['pid'].cat.codes.map(lambda x: palette[x % len(palette)])

        if dims == 2:
            create_plot_2d(df)
            df.to_csv(save_path + f'{reduction_method}_2d.csv', index=False)
        elif dims == 3:
            create_plot_3d(df)
            df.to_csv(save_path + f'{reduction_method}_3d.csv', index=False)
        else:
            raise ValueError("dims must be 2 or 3")

    def compute(self):
        """
        Computes the CMC and mAP values for the updated features, person identifiers, and camera identifiers.

        Returns:
            np.array, float: An array representing the CMC for each query and the mAP.
        """
        
        if self.feat_norm == 'yes':
            feats = torch.nn.functional.normalize(self.feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        
        m, n = qf.shape[0], gf.shape[0]

        # calculating distance matrix
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
        distmat = distmat.cpu().numpy()

        # PRINT OUT EVERYTHING
        # print("qf", qf.shape)
        # print("gf", gf.shape)
        # print("q_pids", q_pids.shape)
        # print("q_camids", q_camids.shape)
        # print("g_pids", g_pids.shape)
        # print("g_camids", g_camids.shape)
        # print("distmat", distmat.shape)
        
        # calculation cmc and mAP
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP
