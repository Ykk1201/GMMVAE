# GMMVAE for Gene-Promoter Data Clustering

This project uses a Gaussian Mixture Variational Autoencoder (GMMVAE) to perform unsupervised clustering on a gene-promoter matrix. The model learns a low-dimensional representation of the data and clusters the samples into a specified number of groups.

## Dependencies

The following Python libraries are required to run this project:

-   `numpy`
-   `pandas`
-   `torch`
-   `scikit-learn`
-   `tqdm`
-   `umap-learn` (optional, for 2D embedding visualization)

You can install these dependencies using pip:

```bash
pip install numpy pandas torch scikit-learn tqdm umap-learn
```

## Data Format

The input data should be a text file (e.g., CSV or TSV) containing a gene-promoter matrix. By default, the script expects a tab-separated file where rows are genes and columns are samples. The first row should be a header with sample names, and the first column should be an index with gene names.

Example `gene_promoter_matrix.txt`:

```
gene    sample1 sample2 sample3
geneA   0.1     1.5     0.8
geneB   2.3     0.2     1.1
geneC   0.5     0.9     2.5
...
```

## Usage

To train the GMMVAE model, run the `train.py` script with the following command-line arguments:

```bash
python train.py --data <path_to_your_data> [options]
```

### Arguments

-   `--data`, `-d`: (Required) Path to the gene promoter matrix file.
-   `--outdir`, `-o`: Output directory for the results (default: `output_gmmvae`).
-   `--sep`: File delimiter for the input data (default: `\t`).
-   `--batch_size`: Batch size for training (default: `128`).
-   `--epochs`: Number of training epochs (default: `200`).
-   `--lr`: Learning rate (default: `2e-4`).
-   `--weight_decay`: Weight decay for the optimizer (default: `5e-4`).
-   `--latent_dim`: Dimensionality of the latent space (default: `10`).
-   `--n_clusters`, `-k`: Number of clusters for the GMM (default: `10`).
-   `--encode_dim`: Comma-separated list of hidden layer dimensions for the encoder (default: `512,128`).
-   `--decode_dim`: Comma-separated list of hidden layer dimensions for the decoder (default: `128,512`).
-   `--binary`: Use Binary Cross-Entropy for reconstruction (for binary input data).
-   `--log1p`: Apply `log(1+x)` transformation to the input data.
-   `--min_sample_sum`: Minimum sum of values for a sample to be included (default: `0.0`).
-   `--min_gene_var`: Minimum variance for a gene to be included (default: `0.0`).
-   `--seed`: Random seed for reproducibility (default: `18`).
-   `--gpu`: GPU device to use (default: `0`).
-   `--cpu`: Force to use CPU even if a GPU is available.

## Output

The script will create the specified output directory and save the following files:

-   `embedding.npy`: The learned latent space embeddings for each sample.
-   `cluster_probabilities.npy`: The probability of each sample belonging to each cluster.
-   `cluster_assignment.csv`: A CSV file with the cluster assignment for each sample.
-   `embedding.csv`: The latent space embeddings in CSV format, with sample names.
-   `embedding_2d_umap.csv` or `embedding_2d_pca.csv`: A 2D representation of the embeddings for visualization (using UMAP if available, otherwise PCA).
-   `model.pt`: The saved PyTorch model state dictionary.
-   `run_config.json`: A JSON file with the command-line arguments used for the run.
-   `genes_used.txt`: A list of the genes used for training after filtering.


## replication: run this commend for exact replication
## hyperparameter are set for optimal clustering performance based on experimentation
python train.py --data gene_promoter_matrix.txt --log1p --min_gene_var 0.01 \
    --n_clusters 15 --latent_dim 10 --epochs 200 --kl_anneal_epochs 50