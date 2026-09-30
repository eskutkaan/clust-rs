//! Consensus clustering across multiple K values.
//!
//! Pipeline per K:
//! 1. Cluster the matrix for each requested K.
//! 2. Relabel partitions to a common reference.
//! 3. Combine memberships into consensus candidates.

use crate::cluster::{
    expand_membership, generate_partitions_for_matrix, kmeans, labels_to_membership, Membership,
};
use crate::data::subset_present;
use ndarray::Array2;
use rayon::prelude::*;

/// Greedy min-min relabel of `other` columns to best match `reference`.
pub fn relabel_to_ref(reference: &Membership, other: &Membership) -> Membership {
    let k_ref = reference.ncols();
    let k_other = other.ncols();
    let n = reference.nrows().min(other.nrows());
    let k = k_ref.min(k_other);

    let mut dist = Array2::<f64>::zeros((k_ref, k_other));
    for i in 0..k_ref {
        for j in 0..k_other {
            let d: f64 = (0..n)
                .map(|r| {
                    let a = reference[[r, i]];
                    let b = other[[r, j]];
                    (a - b).powi(2)
                })
                .sum();
            dist[[i, j]] = d;
        }
    }

    let mut perm = vec![usize::MAX; k_other];
    let mut used_ref = vec![false; k_ref];
    let mut used_other = vec![false; k_other];

    for _ in 0..k {
        let mut best = (0usize, 0usize, f64::INFINITY);
        for i in 0..k_ref {
            if used_ref[i] {
                continue;
            }
            for j in 0..k_other {
                if used_other[j] {
                    continue;
                }
                if dist[[i, j]] < best.2 {
                    best = (i, j, dist[[i, j]]);
                }
            }
        }
        if best.2.is_infinite() {
            break;
        }
        perm[best.1] = best.0;
        used_ref[best.0] = true;
        used_other[best.1] = true;
    }

    let mut next = 0;
    for j in 0..k_other {
        if perm[j] == usize::MAX {
            while next < k_ref && used_ref[next] {
                next += 1;
            }
            if next < k_ref {
                perm[j] = next;
                used_ref[next] = true;
            } else {
                perm[j] = j % k_ref.max(1);
            }
        }
    }

    let mut out = Array2::<f64>::zeros((reference.nrows(), k_ref));
    for j in 0..k_other {
        let target = if k_ref == 0 {
            0
        } else {
            perm[j].min(k_ref - 1)
        };
        for i in 0..other.nrows().min(reference.nrows()) {
            out[[i, target]] += other[[i, j]];
        }
    }
    out
}

/// Build a consensus membership matrix for one K.
pub fn copam_for_k(
    matrices: &[Array2<f64>],
    gdm: &Array2<bool>,
    k: usize,
    n_init: usize,
    max_iter: usize,
    seed: u64,
) -> Membership {
    let n_genes = gdm.nrows();
    let n_ds = matrices.len();
    if n_ds == 0 || k == 0 {
        return Array2::zeros((n_genes, 0));
    }

    // Cluster each matrix partition and expand it to the global gene universe.
    let per_ds: Vec<Membership> = (0..n_ds)
        .into_par_iter()
        .map(|li| {
            let gdm_col: Vec<bool> = (0..n_genes).map(|i| gdm[[i, li]]).collect();
            let (sub, local_idx) = subset_present(&matrices[li], &gdm_col);
            if sub.nrows() < k {
                return Array2::zeros((n_genes, k));
            }
            let (labels, _) = kmeans(
                &sub,
                k,
                max_iter,
                n_init,
                seed.wrapping_add((li as u64 + 1) * 10007 + k as u64),
            );
            let local_mem = labels_to_membership(&labels, k);
            expand_membership(&local_mem, &local_idx, n_genes)
        })
        .collect();

    // Use the first partition as the reference for relabeling.
    let mut ref_mem = per_ds[0].clone();
    let mut weight = Array2::<f64>::zeros((n_genes, k));
    let mut sum = Array2::<f64>::zeros((n_genes, k));

    for (li, mem) in per_ds.iter().enumerate() {
        let aligned = if li == 0 {
            mem.clone()
        } else {
            relabel_to_ref(&ref_mem, mem)
        };
        if li == 0 {
            ref_mem = aligned.clone();
        }
        for i in 0..n_genes {
            if !gdm[[i, li]] {
                continue;
            }
            for c in 0..k {
                sum[[i, c]] += aligned[[i, c]];
                weight[[i, c]] += 1.0;
            }
        }
    }

    for i in 0..n_genes {
        for c in 0..k {
            if weight[[i, c]] > 0.0 {
                sum[[i, c]] /= weight[[i, c]];
            }
        }
    }
    sum
}

/// Run UNCLES-style consensus over a range of K values in parallel.
pub fn uncles_consensus(
    matrices: &[Array2<f64>],
    gdm: &Array2<bool>,
    ks: &[usize],
    n_init: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<(usize, Membership)> {
    ks.par_iter()
        .map(|&k| {
            let copam = copam_for_k(matrices, gdm, k, n_init, max_iter, seed);
            (k, copam)
        })
        .collect()
}

/// Difference-threshold binarisation (DTB): assign gene to top cluster if
/// membership exceeds second by `diff`.

/// Assign gene to every cluster with membership >= threshold (softer than DTB).
pub fn binarise_threshold(mem: &Membership, threshold: f64) -> Array2<bool> {
    let n = mem.nrows();
    let k = mem.ncols();
    let mut b = Array2::<bool>::from_elem((n, k), false);
    for i in 0..n {
        for c in 0..k {
            if mem[[i, c]] >= threshold {
                b[[i, c]] = true;
            }
        }
    }
    b
}

pub fn binarise_dtb(mem: &Membership, diff: f64) -> Array2<bool> {
    let n = mem.nrows();
    let k = mem.ncols();
    let mut b = Array2::<bool>::from_elem((n, k), false);
    for i in 0..n {
        let row = mem.row(i);
        let mut vals: Vec<(usize, f64)> = row.iter().copied().enumerate().collect();
        vals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if vals.is_empty() {
            continue;
        }
        let (best_c, best_v) = vals[0];
        let second = if vals.len() > 1 { vals[1].1 } else { 0.0 };
        if best_v - second >= diff && best_v > 0.0 {
            b[[i, best_c]] = true;
        }
    }
    b
}

/// Stack binary cluster columns from many CoPaMs into one big candidate matrix.
pub fn collect_binary_candidates(copams: &[(usize, Membership)], diff: f64) -> Array2<bool> {
    let bins: Vec<Array2<bool>> = copams
        .iter()
        .flat_map(|(_, m)| vec![binarise_dtb(m, diff), binarise_threshold(m, 0.5)])
        .collect();
    if bins.is_empty() {
        return Array2::from_elem((0, 0), false);
    }
    let n = bins[0].nrows();
    let total_k: usize = bins.iter().map(|b| b.ncols()).sum();
    let mut out = Array2::<bool>::from_elem((n, total_k), false);
    let mut col = 0;
    for b in &bins {
        for c in 0..b.ncols() {
            for i in 0..n {
                out[[i, col]] = b[[i, c]];
            }
            col += 1;
        }
    }
    out
}

/// Single-dataset fallback: multi-K k-means consensus (no GDM needed).
pub fn single_dataset_consensus(
    data: &Array2<f64>,
    ks: &[usize],
    n_init: usize,
    max_iter: usize,
    seed: u64,
) -> Membership {
    let parts = generate_partitions_for_matrix(data, ks, n_init, max_iter, seed);
    if parts.is_empty() {
        return Array2::zeros((data.nrows(), 0));
    }
    let mut sorted = parts;
    sorted.sort_by_key(|(k, _)| *k);
    let (_, ref_mem) = &sorted[sorted.len() / 2];
    let k_ref = ref_mem.ncols();
    let n = ref_mem.nrows();
    let mut copam = Array2::<f64>::zeros((n, k_ref));
    let mut w = 0.0f64;
    for (_, mem) in &sorted {
        let aligned = if mem.ncols() == k_ref {
            relabel_to_ref(ref_mem, mem)
        } else {
            continue;
        };
        for i in 0..n {
            for c in 0..k_ref {
                copam[[i, c]] += aligned[[i, c]];
            }
        }
        w += 1.0;
    }
    if w > 0.0 {
        copam.mapv_inplace(|v| v / w);
    }
    copam
}
