#!/usr/bin/env python
"""Train GMMVAE on a gene-promoter matrix and export embeddings/clusters."""

import argparse
import json
import os
import random
import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from tqdm import tqdm


class GenePromoterDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = np.asarray(x, dtype=np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx]


def load_gene_promoter_matrix(
    path: str,
    sep: str = "\t",
    index_col: int = 0,
    gene_by_sample: bool = True,
    log1p: bool = False,
    min_sample_sum: float = 0.0,
    min_gene_var: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=sep, header=0, index_col=index_col)
    if gene_by_sample:
        df = df.T  # -> sample x gene

    x = df.to_numpy(dtype=np.float32)
    if log1p:
        x = np.log1p(x)

    if min_sample_sum > 0:
        keep_sample = np.sum(x, axis=1) >= float(min_sample_sum)
        x = x[keep_sample]
        df = df.iloc[keep_sample]

    if min_gene_var > 0:
        keep_gene = np.var(x, axis=0) >= float(min_gene_var)
        x = x[:, keep_gene]
        df = df.iloc[:, keep_gene]

    sample_names = df.index.to_numpy()
    gene_names = df.columns.to_numpy()
    return x.astype(np.float32), sample_names, gene_names


def create_dataloaders(x: np.ndarray, batch_size: int = 128, num_workers: int = 0):
    dataset = GenePromoterDataset(x)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, full_loader


def build_mlp(layers, activation=nn.ReLU(), bn=False, dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if bn:
            net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class GaussianSample(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        epsilon = torch.randn(mu.size(), requires_grad=False, device=mu.device)
        std = log_var.mul(0.5).exp_()
        z = mu.addcmul(std, epsilon)
        return z, mu, log_var


class Encoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0):
        super().__init__()
        x_dim, h_dim, z_dim = dims
        self.hidden = build_mlp([x_dim] + h_dim, bn=bn, dropout=dropout)
        self.sample = GaussianSample(([x_dim] + h_dim)[-1], z_dim)

    def forward(self, x):
        return self.sample(self.hidden(x))


class Decoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0, output_activation=None):
        super().__init__()
        z_dim, h_dim, x_dim = dims
        self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
        self.reconstruction = nn.Linear([z_dim, *h_dim][-1], x_dim)
        self.output_activation = output_activation

    def forward(self, x):
        x = self.hidden(x)
        x = self.reconstruction(x)
        return self.output_activation(x) if self.output_activation is not None else x


def reconstruction_loss(recon_x: torch.Tensor, x: torch.Tensor, binary: bool = False) -> torch.Tensor:
    if binary:
        return F.binary_cross_entropy(recon_x, x, reduction="none").sum(dim=1)
    return F.mse_loss(recon_x, x, reduction="none").sum(dim=1)


def gmmvae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    gamma: torch.Tensor,
    mu_c: torch.Tensor,
    var_c: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    binary: bool = False,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-8
    var_c = var_c + eps
    gamma = gamma.clamp(min=eps)
    pi = pi.clamp(min=eps)

    n_centroids = pi.size(1)
    mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
    logvar_expand = logvar.unsqueeze(2).expand(logvar.size(0), logvar.size(1), n_centroids)

    recon = reconstruction_loss(recon_x, x, binary=binary)

    logpzc = -0.5 * torch.sum(
        gamma
        * torch.sum(
            math.log(2 * math.pi)
            + torch.log(var_c)
            + torch.exp(logvar_expand) / var_c
            + (mu_expand - mu_c) ** 2 / var_c,
            dim=1,
        ),
        dim=1,
    )
    logpc = torch.sum(gamma * torch.log(pi), dim=1)
    qentropy = -0.5 * torch.sum(1 + logvar + math.log(2 * math.pi), dim=1)
    logqcx = torch.sum(gamma * torch.log(gamma), dim=1)

    kl = -logpzc - logpc + qentropy + logqcx
    total = recon + kl_weight * kl
    return total.mean(), recon.mean(), kl.mean()


class GMMVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        n_clusters: int = 10,
        encode_dim: Optional[list[int]] = None,
        decode_dim: Optional[list[int]] = None,
        binary: bool = False,
        dropout: float = 0.0,
        bn: bool = False,
    ):
        super().__init__()
        encode_dim = encode_dim or [512, 128]
        decode_dim = decode_dim or [128, 512]
        self.binary = binary
        self.n_clusters = n_clusters

        decode_activation = nn.Sigmoid() if binary else None
        self.encoder = Encoder([input_dim, encode_dim, latent_dim], bn=bn, dropout=dropout)
        self.decoder = Decoder([latent_dim, decode_dim, input_dim], bn=bn, dropout=dropout, output_activation=decode_activation)

        self.pi = nn.Parameter(torch.ones(n_clusters) / n_clusters)
        self.mu_c = nn.Parameter(torch.zeros(latent_dim, n_clusters))
        self.log_var_c = nn.Parameter(torch.zeros(latent_dim, n_clusters))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_gamma(self, z: torch.Tensor):
        eps = 1e-8
        n = z.size(0)
        k = self.n_clusters
        z_expand = z.unsqueeze(2).expand(n, z.size(1), k)

        pi = F.softmax(self.pi, dim=0).clamp(min=eps).repeat(n, 1)
        mu_c = self.mu_c.repeat(n, 1, 1)
        var_c = self.log_var_c.exp().repeat(n, 1, 1)

        p_c_z = torch.exp(
            torch.log(pi) - torch.sum(0.5 * torch.log(2 * math.pi * var_c) + (z_expand - mu_c) ** 2 / (2 * var_c), dim=1)
        )
        p_c_z = p_c_z + eps
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)
        return gamma, mu_c, var_c, pi

    def forward(self, x: torch.Tensor):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        gamma, mu_c, var_c, pi = self.get_gamma(z)
        return recon_x, z, mu, logvar, gamma, mu_c, var_c, pi

    def loss_function(self, x: torch.Tensor, kl_weight: float = 1.0):
        recon_x, _, mu, logvar, gamma, mu_c, var_c, pi = self.forward(x)
        return gmmvae_loss(recon_x, x, gamma, mu_c, var_c, pi, mu, logvar, binary=self.binary, kl_weight=kl_weight)

    @torch.no_grad()
    def encode_batch(self, dataloader, device: str = "cpu", out: str = "mu") -> np.ndarray:
        self.eval()
        outputs = []
        for x in dataloader:
            x = x.float().to(device)
            z, mu, _ = self.encoder(x)
            outputs.append(z.cpu() if out == "z" else mu.cpu())
        return torch.cat(outputs, dim=0).numpy()

    @torch.no_grad()
    def predict_clusters(self, dataloader, device: str = "cpu", use: str = "mu"):
        self.eval()
        all_gamma = []
        for x in dataloader:
            x = x.float().to(device)
            z, mu, _ = self.encoder(x)
            feat = mu if use == "mu" else z
            gamma, _, _, _ = self.get_gamma(feat)
            all_gamma.append(gamma.cpu())
        gamma = torch.cat(all_gamma, dim=0).numpy()
        labels = np.argmax(gamma, axis=1)
        return labels, gamma

    @torch.no_grad()
    def initialize_gmm_params(self, dataloader, device: str = "cpu", use: str = "mu"):
        feat = self.encode_batch(dataloader, device=device, out=use)
        gmm = GaussianMixture(n_components=self.n_clusters, covariance_type="diag", random_state=0)
        gmm.fit(feat)
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)).to(self.mu_c.device))
        log_var = np.log(gmm.covariances_.T.clip(min=1e-6)).astype(np.float32)
        self.log_var_c.data.copy_(torch.from_numpy(log_var).to(self.log_var_c.device))
        self.pi.data.copy_(torch.from_numpy(gmm.weights_.astype(np.float32)).to(self.pi.device))

    @torch.no_grad()
    def reinit_dead_clusters(self, dataloader, device: str = "cpu", min_cluster_size: int = 5):
        """Re-seed centroids of dead clusters to random high-density embedding regions."""
        self.eval()
        all_mu, all_gamma = [], []
        for x in dataloader:
            x = x.float().to(device)
            _, mu, _ = self.encoder(x)
            gamma, _, _, _ = self.get_gamma(mu)
            all_mu.append(mu.cpu())
            all_gamma.append(gamma.cpu())
        all_mu = torch.cat(all_mu, dim=0)       # (N, latent_dim)
        all_gamma = torch.cat(all_gamma, dim=0) # (N, n_clusters)

        cluster_counts = all_gamma.argmax(dim=1).bincount(minlength=self.n_clusters)
        dead = (cluster_counts < min_cluster_size).nonzero(as_tuple=True)[0]

        if len(dead) == 0:
            return 0

        alive = (cluster_counts >= min_cluster_size).nonzero(as_tuple=True)[0]
        if len(alive) > 0:
            mean_log_var = self.log_var_c.data[:, alive].mean(dim=1, keepdim=True)
        else:
            mean_log_var = torch.zeros(self.log_var_c.size(0), 1, device=self.log_var_c.device)

        perm = torch.randperm(all_mu.size(0))
        for i, c in enumerate(dead):
            src = all_mu[perm[i % all_mu.size(0)]]
            self.mu_c.data[:, c] = src.to(self.mu_c.device)
            self.log_var_c.data[:, c] = mean_log_var.squeeze(1).to(self.log_var_c.device)

        return len(dead)

    def fit(
        self,
        dataloader,
        device: str = "cpu",
        lr: float = 2e-4,
        weight_decay: float = 5e-4,
        max_epochs: int = 200,
        kl_anneal_epochs: int = 50,
        kl_weight_max: float = 0.5,
        grad_clip: float = 10.0,
        reinit_interval: int = 10,
        min_cluster_size: int = 5,
        verbose: bool = True,
    ):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        epoch_iter = tqdm(range(max_epochs), total=max_epochs, desc="Epoch") if verbose else range(max_epochs)
        for epoch in epoch_iter:
            self.train()
            total_loss = total_recon = total_kl = 0.0
            n_batches = 0

            if kl_anneal_epochs > 0 and epoch < kl_anneal_epochs:
                kl_weight = kl_weight_max * epoch / kl_anneal_epochs
            else:
                kl_weight = kl_weight_max

            for x in dataloader:
                x = x.float().to(device)
                optimizer.zero_grad()
                loss, recon, kl = self.loss_function(x, kl_weight=kl_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()
                n_batches += 1

            # Re-seed dead clusters periodically (only during KL annealing phase)
            n_reinit = 0
            if reinit_interval > 0 and epoch < kl_anneal_epochs and (epoch + 1) % reinit_interval == 0:
                n_reinit = self.reinit_dead_clusters(dataloader, device=device, min_cluster_size=min_cluster_size)

            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(
                    loss=f"{total_loss / max(n_batches, 1):.4f}",
                    recon=f"{total_recon / max(n_batches, 1):.4f}",
                    kl=f"{total_kl / max(n_batches, 1):.4f}",
                    kl_w=f"{kl_weight:.2f}",
                    dead=n_reinit,
                )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_hidden_dims(text: str) -> list[int]:
    if text.strip() == "":
        return []
    return [int(x) for x in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Train GMMVAE for unsupervised clustering")
    parser.add_argument("--data", "-d", type=str, required=True, help="Path to gene promoter matrix (gene x sample)")
    parser.add_argument("--outdir", "-o", type=str, default="output_gmmvae", help="Output directory")
    parser.add_argument("--sep", type=str, default="\t", help="File delimiter, default: tab")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)    
    parser.add_argument("--kl_anneal_epochs", type=int, default=50, help="Epochs to anneal KL weight")
    parser.add_argument("--kl_weight_max", type=float, default=0.5, help="Max KL weight after annealing (beta). <1 keeps latent space more discriminative")
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--n_clusters", "-k", type=int, default=15)
    parser.add_argument("--reinit_interval", type=int, default=10, help="Re-seed dead clusters every N epochs (0=off)")
    parser.add_argument("--min_cluster_size", type=int, default=5, help="Clusters below this count get re-seeded")
    parser.add_argument("--encode_dim", type=str, default="512,128", help="Comma separated hidden dims")
    parser.add_argument("--decode_dim", type=str, default="128,512", help="Comma separated hidden dims")

    parser.add_argument("--binary", action="store_true", help="Use BCE reconstruction (for binary inputs)")
    parser.add_argument("--log1p", action="store_true", help="Apply log1p to matrix values")
    parser.add_argument("--min_sample_sum", type=float, default=0.0)
    parser.add_argument("--min_gene_var", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    if torch.cuda.is_available() and not args.cpu:
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"

    x, sample_names, gene_names = load_gene_promoter_matrix(
        path=args.data,
        sep=args.sep,
        gene_by_sample=True,
        log1p=args.log1p,
        min_sample_sum=args.min_sample_sum,
        min_gene_var=args.min_gene_var,
    )

    train_loader, full_loader = create_dataloaders(x, batch_size=args.batch_size)

    model = GMMVAE(
        input_dim=x.shape[1],
        latent_dim=args.latent_dim,
        n_clusters=args.n_clusters,
        encode_dim=parse_hidden_dims(args.encode_dim),
        decode_dim=parse_hidden_dims(args.decode_dim),
        binary=args.binary,
    )

    model.to(device)
    model.initialize_gmm_params(full_loader, device=device, use="mu")
    model.fit(
        train_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        kl_anneal_epochs=args.kl_anneal_epochs,
        kl_weight_max=args.kl_weight_max,
        reinit_interval=args.reinit_interval,
        min_cluster_size=args.min_cluster_size,
        verbose=True,
    )

    embedding = model.encode_batch(full_loader, device=device, out="mu")
    labels, probs = model.predict_clusters(full_loader, device=device, use="mu")

    np.save(os.path.join(args.outdir, "embedding.npy"), embedding)
    np.save(os.path.join(args.outdir, "cluster_probabilities.npy"), probs)

    cluster_df = pd.DataFrame(
        {
            "sample": sample_names,
            "cluster": labels,
        }
    )
    cluster_df.to_csv(os.path.join(args.outdir, "cluster_assignment.csv"), index=False)

    emb_df = pd.DataFrame(embedding)
    emb_df.insert(0, "sample", sample_names)
    emb_df.to_csv(os.path.join(args.outdir, "embedding.csv"), index=False)

    # 2D embedding for visualization
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=args.seed)
        emb2d = reducer.fit_transform(embedding)
        emb2d_method = "umap"
    except Exception:
        emb2d = PCA(n_components=2, random_state=args.seed).fit_transform(embedding)
        emb2d_method = "pca"

    emb2d_df = pd.DataFrame(
        {
            "sample": sample_names,
            "dim1": emb2d[:, 0],
            "dim2": emb2d[:, 1],
            "cluster": labels,
        }
    )
    emb2d_df.to_csv(os.path.join(args.outdir, f"embedding_2d_{emb2d_method}.csv"), index=False)

    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pt"))

    with open(os.path.join(args.outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    pd.Series(gene_names).to_csv(os.path.join(args.outdir, "genes_used.txt"), index=False, header=False)

    print(f"Training done. samples={x.shape[0]}, genes={x.shape[1]}, output={args.outdir}")


if __name__ == "__main__":
    main()
