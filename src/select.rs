//! M-N style cluster selection.
use crate::cluster::cluster_mse;
use ndarray::Array2;
use rayon::prelude::*;

/// Calculate the mean squared error for a cluster mask.
fn matrix_mse(matrices: &[Array2<f64>], gdm: &Array2<bool>, mask: &[bool]) -> f64 {
    let mut mses = Vec::new();
    for (li, mat) in matrices.iter().enumerate() {
        // Restrict the mask to genes present in this matrix.
        let local_mask: Vec<bool> = mask
            .iter()
            .enumerate()
            .map(|(i, &m)| m && gdm[[i, li]])
            .collect();
        let n_local = local_mask.iter().filter(|&&m| m).count();
        if n_local == 0 {
            continue;
        }
        // Build the cluster submatrix for the MSE calculation.
        // cluster_mse expects mask aligned with mat rows
        let mse = cluster_mse(mat, &local_mask);
        if mse.is_finite() {
            mses.push(mse);
        }
    }
    if mses.is_empty() {
        f64::NAN
    } else {
        mses.iter().sum::<f64>() / mses.len() as f64
    }
}

/// Select non-overlapping tight clusters (M-N greedy).
pub fn select_clusters(
    matrices: &[Array2<f64>],
    gdm: &Array2<bool>,
    binary: &Array2<bool>,
    tightness_weight: f64,
    min_size: usize,
    max_clusters: Option<usize>,
) -> Array2<bool> {
    let k = binary.ncols();
    let n = binary.nrows();
    if k == 0 || n == 0 {
        return Array2::from_elem((n, 0), false);
    }

    #[derive(Clone)]
    struct Cand {
        size: usize,
        mse: f64,
        dist: f64,
        mask: Vec<bool>,
    }

    let mut cands: Vec<Cand> = (0..k)
        .into_par_iter()
        .filter_map(|c| {
            let mask: Vec<bool> = (0..n).map(|i| binary[[i, c]]).collect();
            let size = mask.iter().filter(|&&m| m).count();
            if size < min_size {
                return None;
            }
            let mse = matrix_mse(matrices, gdm, &mask);
            if !mse.is_finite() {
                return None;
            }
            Some(Cand {
                size,
                mse,
                dist: 0.0,
                mask,
            })
        })
        .collect();

    if cands.is_empty() {
        return Array2::from_elem((n, 0), false);
    }

    let max_mse = cands
        .iter()
        .map(|c| c.mse)
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-12);
    let max_log_size = (cands.iter().map(|c| c.size).max().unwrap() as f64)
        .log10()
        .max(1e-12);

    for c in &mut cands {
        let x = (c.mse / max_mse) * tightness_weight;
        let y = 1.0 - (c.size as f64).log10() / max_log_size;
        c.dist = (x * x + y * y).sqrt();
    }
    cands.sort_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut taken = vec![false; n];
    let mut selected: Vec<Vec<bool>> = Vec::new();
    for cand in &cands {
        if max_clusters.is_some_and(|limit| selected.len() >= limit) {
            break;
        }
        if cand.mask.iter().enumerate().any(|(i, &m)| m && taken[i]) {
            continue;
        }
        for (i, &m) in cand.mask.iter().enumerate() {
            if m {
                taken[i] = true;
            }
        }
        selected.push(cand.mask.clone());
    }

    if selected.is_empty() {
        return Array2::from_elem((n, 0), false);
    }
    let n_clust = selected.len();
    let mut out = Array2::<bool>::from_elem((n, n_clust), false);
    for (c, mask) in selected.into_iter().enumerate() {
        for i in 0..n {
            out[[i, c]] = mask[i];
        }
    }
    out
}
