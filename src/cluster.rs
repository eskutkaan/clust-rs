//! Parallel k-means (ndarray + rayon).
use ndarray::{Array2, Axis};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub type Labels = Vec<usize>;
pub type Membership = Array2<f64>;

fn euclidean_sq(a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn init_plus_plus(data: &Array2<f64>, k: usize, rng: &mut ChaCha8Rng) -> Array2<f64> {
    let n = data.nrows();
    let dim = data.ncols();
    let mut centres = Array2::<f64>::zeros((k, dim));
    let first = rng.gen_range(0..n);
    centres.row_mut(0).assign(&data.row(first));
    let mut min_dist = vec![f64::INFINITY; n];
    for c in 1..k {
        min_dist.par_iter_mut().enumerate().for_each(|(i, d)| {
            let dist = euclidean_sq(data.row(i), centres.row(c - 1));
            if dist < *d {
                *d = dist;
            }
        });
        let total: f64 = min_dist.iter().sum();
        if total <= 0.0 {
            let idx = rng.gen_range(0..n);
            centres.row_mut(c).assign(&data.row(idx));
            continue;
        }
        let mut r = rng.gen::<f64>() * total;
        let mut chosen = 0usize;
        for i in 0..n {
            r -= min_dist[i];
            if r <= 0.0 {
                chosen = i;
                break;
            }
        }
        centres.row_mut(c).assign(&data.row(chosen));
    }
    centres
}

pub fn kmeans(
    data: &Array2<f64>,
    k: usize,
    max_iter: usize,
    n_init: usize,
    seed: u64,
) -> (Labels, f64) {
    let n = data.nrows();
    if n == 0 || k == 0 {
        return (vec![], f64::INFINITY);
    }
    let k = k.min(n);
    let mut best_labels = vec![0usize; n];
    let mut best_inertia = f64::INFINITY;

    for init_id in 0..n_init {
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(init_id as u64));
        let mut centres = init_plus_plus(data, k, &mut rng);
        let mut labels = vec![0usize; n];

        for _ in 0..max_iter {
            let changed = AtomicUsize::new(0);
            labels.par_iter_mut().enumerate().for_each(|(i, lab)| {
                let row = data.row(i);
                let mut best = 0usize;
                let mut best_d = f64::INFINITY;
                for c in 0..k {
                    let d = euclidean_sq(row, centres.row(c));
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                if *lab != best {
                    changed.fetch_add(1, Ordering::Relaxed);
                    *lab = best;
                }
            });
            if changed.load(Ordering::Relaxed) == 0 {
                break;
            }

            let mut counts = vec![0usize; k];
            let mut new_centres = Array2::<f64>::zeros((k, data.ncols()));
            for i in 0..n {
                let c = labels[i];
                counts[c] += 1;
                let row = data.row(i);
                for j in 0..data.ncols() {
                    new_centres[[c, j]] += row[j];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    let inv = 1.0 / counts[c] as f64;
                    new_centres.row_mut(c).mapv_inplace(|v| v * inv);
                } else {
                    let idx = rng.gen_range(0..n);
                    new_centres.row_mut(c).assign(&data.row(idx));
                }
            }
            centres = new_centres;
        }

        let inertia: f64 = labels
            .par_iter()
            .enumerate()
            .map(|(i, &c)| euclidean_sq(data.row(i), centres.row(c)))
            .sum();
        if inertia < best_inertia {
            best_inertia = inertia;
            best_labels = labels;
        }
    }
    (best_labels, best_inertia)
}

pub fn labels_to_membership(labels: &[usize], k: usize) -> Membership {
    let n = labels.len();
    let mut m = Array2::<f64>::zeros((n, k));
    for (i, &lab) in labels.iter().enumerate() {
        if lab < k {
            m[[i, lab]] = 1.0;
        }
    }
    m
}

/// Scatter membership of present genes back into full n_genes × K matrix.
pub fn expand_membership(
    local: &Membership,
    local_to_global: &[usize],
    n_global: usize,
) -> Membership {
    let k = local.ncols();
    let mut full = Array2::<f64>::zeros((n_global, k));
    for (li, &gi) in local_to_global.iter().enumerate() {
        for c in 0..k {
            full[[gi, c]] = local[[li, c]];
        }
    }
    full
}

pub fn generate_partitions_for_matrix(
    data: &Array2<f64>,
    ks: &[usize],
    n_init: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<(usize, Membership)> {
    ks.par_iter()
        .map(|&k| {
            let (labels, _) = kmeans(data, k, max_iter, n_init, seed.wrapping_add(k as u64));
            let mem = labels_to_membership(&labels, k);
            (k, mem)
        })
        .collect()
}

pub fn cluster_mse(data: &Array2<f64>, mask: &[bool]) -> f64 {
    let members: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();
    if members.is_empty() {
        return f64::NAN;
    }
    let sub = data.select(Axis(0), &members);
    let centre = sub.mean_axis(Axis(0)).unwrap();
    let mut sse = 0.0;
    for i in 0..sub.nrows() {
        sse += euclidean_sq(sub.row(i), centre.view());
    }
    sse / (sub.nrows() as f64 * sub.ncols() as f64)
}

pub fn cluster_sizes(binary: &Array2<bool>) -> Vec<usize> {
    (0..binary.ncols())
        .map(|c| binary.column(c).iter().filter(|&&v| v).count())
        .collect()
}
