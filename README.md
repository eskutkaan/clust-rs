# Clust-RS

Clust-RS is a Rust command-line program for consensus clustering of a single gene-expression matrix. It normalises and filters expression data, performs k-means clustering across multiple `K` values, selects non-overlapping clusters, and writes TSV summaries plus a PDF report.

## Quick Start

Requirements:

- Rust toolchain with Cargo
- Input data in CSV, TSV, or semicolon-delimited matrix format

Build and run the bundled data example:

```bash
cargo run -- data/example.csv -o data_results -j 0
```

For a faster small example:

```bash
cargo run -- example_data/demo.tsv -o demo_results -K 4 --cs 2 -j 1 --seed 42
```

The output directory contains:

- `Clusters_Objects.tsv`: selected cluster members
- `Gene_to_cluster.tsv`: gene-to-cluster assignments
- `Summary.tsv`: run parameters and summary statistics
- `Cluster_Expression_Profiles.pdf`: up to three pages containing all cluster expression line plots and heatmaps

## Input Format

Each file must contain a header row. The first column is the gene identifier; all remaining columns are samples:

```text
FeatureName,ctrl,3hpi,12hpi,1dpi,2dpi
GeneA,0.68,0.01,0.01,0.33,0.31
GeneB,0.25,0.60,0.66,0.40,0.14
```

Pass one CSV, TSV, or TXT matrix file:

```bash
cargo run -- data/example.csv -o one_dataset_results
```

## Common Options

```text
-o <dir>         Output directory (default: clust_rs_results)
-n <codes...>    Normalisation codes (default: 1000)
-K <ints...>     K values (default: 4 8 12 16 20)
-t <float>       Tightness weight (default: 1.0)
--cs <int>       Minimum cluster size (default: 11)
--clusters <int> Force the exact number of output clusters
-j <int>         Threads; 0 uses all available CPUs (default: 0)
--seed <int>     Random seed (default: 42)
--diff <float>   Membership gap for candidate binarisation (default: 0.1)
```

For example:

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

## Processing Details

The full algorithm description is in [`docs/PROCESSING.md`](docs/PROCESSING.md). It covers input loading, every supported normalization code, flat-gene filtering, k-means, consensus membership construction, candidate binarisation, M-N-style selection, and output interpretation.

## Development

Run the local checks before opening a pull request:

```bash
cargo check
cargo test
```

The project is Rust-only. The library crate exposes the processing modules for Rust callers, and the `clust-rs` binary provides the command-line interface.
