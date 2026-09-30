//! Normalisation and gene filtering.
use ndarray::{Array2, Axis};

pub fn apply_normalisations(mat: &mut Array2<f64>, codes: &[u32]) {
    for &code in codes {
        match code {
            0 => {}
            1 => row_divide_by_mean(mat),
            3 => log2_safe(mat),
            31 => {
                mat.mapv_inplace(|v| if v < 1.0 { 1.0 } else { v });
                log2_safe(mat);
            }
            4 => row_zscore(mat),
            6 => row_subtract_mean(mat),
            101 => quantile_normalise(mat),
            1000 => auto_normalise(mat),
            _ => {}
        }
    }
}

fn row_stats(mat: &Array2<f64>, row: usize) -> (f64, f64) {
    let r = mat.row(row);
    let valid: Vec<f64> = r.iter().copied().filter(|v| v.is_finite()).collect();
    if valid.is_empty() {
        return (0.0, 1.0);
    }
    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
    if valid.len() < 2 {
        return (mean, 1.0);
    }
    let var = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / valid.len() as f64;
    (mean, var.sqrt().max(1e-12))
}

fn row_divide_by_mean(mat: &mut Array2<f64>) {
    for i in 0..mat.nrows() {
        let (m, _) = row_stats(mat, i);
        if m.abs() > 1e-12 {
            mat.row_mut(i).mapv_inplace(|v| v / m);
        }
    }
}

fn row_subtract_mean(mat: &mut Array2<f64>) {
    for i in 0..mat.nrows() {
        let (m, _) = row_stats(mat, i);
        mat.row_mut(i).mapv_inplace(|v| v - m);
    }
}

fn row_zscore(mat: &mut Array2<f64>) {
    for i in 0..mat.nrows() {
        let (m, s) = row_stats(mat, i);
        mat.row_mut(i).mapv_inplace(|v| (v - m) / s);
    }
}

fn log2_safe(mat: &mut Array2<f64>) {
    mat.mapv_inplace(|v| {
        if !v.is_finite() {
            v
        } else if v <= 0.0 {
            (1e-8f64).log2()
        } else {
            v.log2()
        }
    });
}

fn auto_normalise(mat: &mut Array2<f64>) {
    let positives: Vec<f64> = mat
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if !positives.is_empty() {
        let mut sorted = positives;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if sorted[sorted.len() / 2] > 10.0 {
            log2_safe(mat);
        }
    }
    row_zscore(mat);
}

fn quantile_normalise(mat: &mut Array2<f64>) {
    let n_rows = mat.nrows();
    let n_cols = mat.ncols();
    if n_rows == 0 || n_cols == 0 {
        return;
    }
    let mut col_sorted: Vec<Vec<f64>> = Vec::with_capacity(n_cols);
    for c in 0..n_cols {
        let mut col: Vec<f64> = mat.column(c).iter().copied().collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        col_sorted.push(col);
    }
    let mut rank_means = vec![0.0f64; n_rows];
    for r in 0..n_rows {
        let mut s = 0.0;
        for c in 0..n_cols {
            s += col_sorted[c][r];
        }
        rank_means[r] = s / n_cols as f64;
    }
    for c in 0..n_cols {
        let mut indexed: Vec<(usize, f64)> = mat.column(c).iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, (orig_row, _)) in indexed.into_iter().enumerate() {
            mat[[orig_row, c]] = rank_means[rank];
        }
    }
}

pub fn filter_genes(
    mats: &mut [Array2<f64>],
    gene_ids: &[String],
    gdm: &Array2<bool>,
    min_std: f64,
) -> (Vec<Array2<f64>>, Vec<String>, Array2<bool>, Vec<usize>) {
    let n = gene_ids.len();
    let mut keep = Vec::new();
    'gene: for i in 0..n {
        let mut any = false;
        for (li, mat) in mats.iter().enumerate() {
            if !gdm[[i, li]] {
                continue;
            }
            any = true;
            let row = mat.row(i);
            let vals: Vec<f64> = row.iter().copied().filter(|v| v.is_finite()).collect();
            if vals.len() < 2 {
                continue 'gene;
            }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            if var.sqrt() < min_std {
                continue 'gene;
            }
        }
        if any {
            keep.push(i);
        }
    }
    if keep.is_empty() {
        return (
            vec![],
            vec![],
            Array2::from_elem((0, gdm.ncols()), false),
            vec![],
        );
    }
    let new_mats: Vec<Array2<f64>> = mats.iter().map(|m| m.select(Axis(0), &keep)).collect();
    let new_ids: Vec<String> = keep.iter().map(|&i| gene_ids[i].clone()).collect();
    let new_gdm = gdm.select(Axis(0), &keep);
    (new_mats, new_ids, new_gdm, keep)
}
