# Clust-RS Processing Guide

Clust-RS processes one gene-expression matrix at a time. It normalises and filters the matrix, runs k-means for several requested values of `K`, builds consensus cluster candidates, selects non-overlapping clusters, and writes TSV and PDF results.

## Input

The input must be one CSV, TSV, or semicolon-delimited file:

```text
FeatureName,ctrl,3hpi,12hpi,1dpi,2dpi
GeneA,0.68,0.01,0.01,0.33,0.31
GeneB,0.25,0.60,0.66,0.40,0.14
```

The first column contains gene identifiers. The remaining columns contain sample identifiers and numeric values. Directories are not accepted.

```bash
cargo run -- data/example.csv -o data_results -j 0
cargo run -- example_data/demo.tsv -o demo_results -K 4 --cs 2 -j 1 --seed 42
```

## Normalisation

Non-finite values are replaced with `0.0` before normalisation. Normalisation codes are applied from left to right and can be supplied with `-n`:

| Code | Operation | Description |
| --- | --- | --- |
| `0` | None | Leaves the matrix unchanged. |
| `1` | Row mean division | Divides each gene by its row mean when the mean is non-zero. |
| `3` | Safe log2 | Applies `log2` to positive values; zero and negative values become `log2(1e-8)`. |
| `31` | Floor plus log2 | Floors values below `1.0`, then applies safe log2. |
| `4` | Row z-score | Subtracts each gene's row mean and divides by its row standard deviation. |
| `6` | Row mean subtraction | Subtracts each gene's row mean. |
| `101` | Quantile normalization | Replaces each sample's sorted values with the average value at each rank across samples. |
| `1000` | Automatic normalization | Applies safe log2 when the global positive median exceeds `10`, then applies row z-scoring. |

The default is `1000`.

For gene values $x_1, x_2, \ldots, x_m$, row mean division is:

$$
x'_j = \frac{x_j}{\bar{x}}
$$

Row z-scoring is:

$$
x'_j = \frac{x_j - \bar{x}}{s_x}
$$

## Filtering

After normalisation, genes with insufficient variation are removed. A gene is discarded when its standard deviation is below `1e-6`; rows with fewer than two finite values are also discarded.

## K-Means

For every requested `K`, genes are treated as observations and samples are treated as dimensions:

- Euclidean squared distance is used.
- Initial centers use k-means++-style selection.
- Each run uses up to `100` iterations.
- Three seeded initializations are evaluated.
- The initialization with the lowest within-cluster inertia is retained.

The random seed is controlled by `--seed`. The default `K` values are `4, 8, 12, 16, 20`; override them with `-K`:

```bash
cargo run -- data/example.csv -K 3 5 7 10
```

## Consensus Candidates

Each k-means result is converted into a binary gene-by-cluster membership matrix. Results across the requested `K` values are combined into candidate clusters using two rules:

1. **Difference threshold:** assign a gene to its highest-membership cluster when the gap to its second-highest membership is at least `--diff`.
2. **Fixed threshold:** assign a gene to every cluster with membership at least `0.5`.

The default difference threshold is `0.1`.

## Cluster Selection

Candidate clusters are ranked using within-cluster mean squared error and cluster size. The `--tightness` value controls the weight of the error term. Candidates smaller than `--cs` are discarded.

Candidates are considered from best to worst. A candidate is selected only when none of its genes has already been selected. Final clusters are therefore non-overlapping.

Use `--clusters` to require an exact number of final clusters:

```bash
cargo run -- data/example.csv --clusters 4
```

The program selects the best non-overlapping candidates up to that count and fails if the requested number cannot be formed.

## Outputs

For an output directory such as `data_results/`, the program writes:

| File | Contents |
| --- | --- |
| `Clusters_Objects.tsv` | One row per selected cluster followed by its member genes. |
| `Gene_to_cluster.tsv` | Each retained gene and its first assigned cluster. |
| `Summary.tsv` | Run parameters and summary statistics. |
| `Cluster_Expression_Profiles.pdf` | All selected clusters, with no more than four clusters per page, individual expression profiles, cluster-average lines, and sample-labeled heatmaps without gene-axis labels. |

The PDF uses the normalised and filtered matrix used for clustering.

## CLI Options

```text
clust-rs <matrix-file> [OPTIONS]

-o <dir>         Output directory (default: clust_rs_results)
-n <codes...>    Normalisation codes (default: 1000)
-K <ints...>     K values (default: 4 8 12 16 20)
-t <float>       Tightness weight (default: 1.0)
--cs <int>       Minimum cluster size (default: 11)
--clusters <int> Exact number of output clusters
-j <int>         Threads, 0=all CPUs (default: 0)
--seed <int>     Random seed (default: 42)
--diff <float>   Membership gap for candidate binarisation (default: 0.1)
```

A reproducible run is:

```bash
cargo run -- data/example.csv \
  -o data_results \
  -n 1000 \
  -K 4 8 12 16 20 \
  -t 1.0 \
  --cs 11 \
  -j 0 \
  --seed 42 \
  --diff 0.1
```
