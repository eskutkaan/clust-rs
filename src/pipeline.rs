//! End-to-end single-matrix pipeline.
use crate::cluster::cluster_sizes;
use crate::data::{load_dataset, Dataset};
use crate::error::Result;
use crate::normalise::{apply_normalisations, filter_genes};
use crate::select::select_clusters;
use crate::uncles::{collect_binary_candidates, uncles_consensus};
use ndarray::Array2;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub data_path: PathBuf,
    pub output_dir: PathBuf,
    pub norm_codes: Vec<u32>,
    pub ks: Vec<usize>,
    pub tightness: f64,
    pub min_cluster_size: usize,
    pub forced_clusters: Option<usize>,
    pub n_init: usize,
    pub max_iter: usize,
    pub seed: u64,
    pub binarise_diff: f64,
    pub jobs: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            data_path: PathBuf::from("."),
            output_dir: PathBuf::from("clust_rs_results"),
            norm_codes: vec![1000],
            ks: vec![4, 8, 12, 16, 20],
            tightness: 1.0,
            min_cluster_size: 11,
            forced_clusters: None,
            n_init: 3,
            max_iter: 100,
            seed: 42,
            binarise_diff: 0.1,
            jobs: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PipelineResult {
    pub gene_ids: Vec<String>,
    pub clusters: Array2<bool>,
    pub dataset_names: Vec<String>,
    pub elapsed_secs: f64,
}

pub fn run(cfg: &PipelineConfig) -> Result<PipelineResult> {
    let t0 = Instant::now();
    let n_threads = if cfg.jobs == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        cfg.jobs
    };
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global();

    fs::create_dir_all(&cfg.output_dir)?;

    eprintln!("1. Loading matrix from {}", cfg.data_path.display());
    if !cfg.data_path.is_file() {
        return Err(crate::error::ClustError::InvalidData(
            "Input must be a single matrix file".into(),
        ));
    }
    let dataset = load_dataset(&cfg.data_path)?;
    eprintln!(
        "   {}: {} genes × {} samples",
        dataset.name,
        dataset.n_genes(),
        dataset.n_samples()
    );
    let dataset_names = vec![dataset.name.clone()];
    let gene_ids = dataset.gene_ids.clone();
    let mut matrices = vec![dataset.matrix.clone()];
    let gdm = Array2::from_elem((gene_ids.len(), 1), true);
    let datasets = vec![dataset];

    eprintln!("2. Preparing matrix");
    eprintln!("3. Normalising matrix (codes: {:?})", cfg.norm_codes);
    for mat in matrices.iter_mut() {
        // Replace non-finite values before normalisation.
        mat.mapv_inplace(|v| if v.is_finite() { v } else { 0.0 });
        apply_normalisations(mat, &cfg.norm_codes);
    }

    eprintln!("4. Filtering flat genes");
    let (matrices, gene_ids, gdm, _) = filter_genes(&mut matrices, &gene_ids, &gdm, 1e-6);
    eprintln!("   Genes after filter: {}", gene_ids.len());

    if gene_ids.is_empty() {
        return Ok(PipelineResult {
            gene_ids,
            clusters: Array2::from_elem((0, 0), false),
            dataset_names,
            elapsed_secs: t0.elapsed().as_secs_f64(),
        });
    }

    eprintln!(
        "5. Consensus clustering (K={:?}, threads={})",
        cfg.ks, n_threads
    );
    let copams = uncles_consensus(&matrices, &gdm, &cfg.ks, cfg.n_init, cfg.max_iter, cfg.seed);
    eprintln!("   Built {} CoPaMs", copams.len());

    eprintln!(
        "6. Binarise + M-N select (diff={}, t={}, min_size={})",
        cfg.binarise_diff, cfg.tightness, cfg.min_cluster_size
    );
    let binary = collect_binary_candidates(&copams, cfg.binarise_diff);
    let clusters = select_clusters(
        &matrices,
        &gdm,
        &binary,
        cfg.tightness,
        cfg.min_cluster_size,
        cfg.forced_clusters,
    );
    if let Some(target) = cfg.forced_clusters {
        if clusters.ncols() != target {
            return Err(crate::error::ClustError::InvalidData(format!(
                "Could not select exactly {target} non-overlapping clusters; only {} available",
                clusters.ncols()
            )));
        }
    }
    let sizes = cluster_sizes(&clusters);
    eprintln!("   Selected {} clusters; sizes: {:?}", sizes.len(), sizes);

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("7. Writing results to {}", cfg.output_dir.display());
    write_results(
        &cfg.output_dir,
        &gene_ids,
        &clusters,
        &matrices,
        &gdm,
        &datasets,
        &dataset_names,
        elapsed,
        cfg,
        n_threads,
    )?;

    Ok(PipelineResult {
        gene_ids,
        clusters,
        dataset_names,
        elapsed_secs: elapsed,
    })
}

/// In-memory API: matrices already loaded by a Rust caller.
pub fn run_from_matrices(
    matrices: Vec<Array2<f64>>,
    gene_ids: Vec<String>,
    gdm: Array2<bool>,
    cfg: &PipelineConfig,
) -> Result<PipelineResult> {
    let t0 = Instant::now();
    let n_threads = if cfg.jobs == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        cfg.jobs
    };
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global();

    let mut matrices = matrices;
    for mat in matrices.iter_mut() {
        mat.mapv_inplace(|v| if v.is_finite() { v } else { 0.0 });
        apply_normalisations(mat, &cfg.norm_codes);
    }
    let (matrices, gene_ids, gdm, _) = filter_genes(&mut matrices, &gene_ids, &gdm, 1e-6);
    if gene_ids.is_empty() {
        return Ok(PipelineResult {
            gene_ids,
            clusters: Array2::from_elem((0, 0), false),
            dataset_names: vec![],
            elapsed_secs: t0.elapsed().as_secs_f64(),
        });
    }
    let copams = uncles_consensus(&matrices, &gdm, &cfg.ks, cfg.n_init, cfg.max_iter, cfg.seed);
    let binary = collect_binary_candidates(&copams, cfg.binarise_diff);
    let clusters = select_clusters(
        &matrices,
        &gdm,
        &binary,
        cfg.tightness,
        cfg.min_cluster_size,
        cfg.forced_clusters,
    );
    if let Some(target) = cfg.forced_clusters {
        if clusters.ncols() != target {
            return Err(crate::error::ClustError::InvalidData(format!(
                "Could not select exactly {target} non-overlapping clusters; only {} available",
                clusters.ncols()
            )));
        }
    }
    Ok(PipelineResult {
        gene_ids,
        clusters,
        dataset_names: vec![],
        elapsed_secs: t0.elapsed().as_secs_f64(),
    })
}

fn write_results(
    out: &Path,
    gene_ids: &[String],
    clusters: &Array2<bool>,
    matrices: &[Array2<f64>],
    gdm: &Array2<bool>,
    datasets: &[Dataset],
    dataset_names: &[String],
    elapsed: f64,
    cfg: &PipelineConfig,
    n_threads: usize,
) -> Result<()> {
    let path = out.join("Clusters_Objects.tsv");
    let mut f = File::create(&path)?;
    let n_clust = clusters.ncols();
    for c in 0..n_clust {
        let members: Vec<&str> = (0..clusters.nrows())
            .filter(|&i| clusters[[i, c]])
            .map(|i| gene_ids[i].as_str())
            .collect();
        writeln!(f, "Cluster_{}\t{}", c + 1, members.join("\t"))?;
    }

    let path = out.join("Summary.tsv");
    let mut f = File::create(&path)?;
    writeln!(f, "key\tvalue")?;
    writeln!(f, "n_datasets\t{}", dataset_names.len())?;
    writeln!(f, "datasets\t{}", dataset_names.join(","))?;
    writeln!(f, "n_genes\t{}", gene_ids.len())?;
    writeln!(f, "n_clusters\t{}", n_clust)?;
    writeln!(f, "cluster_sizes\t{:?}", cluster_sizes(clusters))?;
    writeln!(f, "tightness\t{}", cfg.tightness)?;
    writeln!(f, "forced_clusters\t{:?}", cfg.forced_clusters)?;
    writeln!(f, "ks\t{:?}", cfg.ks)?;
    writeln!(f, "norm_codes\t{:?}", cfg.norm_codes)?;
    writeln!(f, "elapsed_seconds\t{:.3}", elapsed)?;
    writeln!(f, "threads\t{}", n_threads)?;
    writeln!(f, "method\tconsensus-k-means")?;

    let path = out.join("Gene_to_cluster.tsv");
    let mut f = File::create(&path)?;
    writeln!(f, "gene_id\tcluster")?;
    for i in 0..gene_ids.len() {
        let mut assigned = false;
        for c in 0..n_clust {
            if clusters[[i, c]] {
                writeln!(f, "{}\t{}", gene_ids[i], c + 1)?;
                assigned = true;
                break;
            }
        }
        if !assigned {
            writeln!(f, "{}\t", gene_ids[i])?;
        }
    }
    write_expression_pdf(out, clusters, matrices, gdm, datasets)?;
    Ok(())
}

fn pdf_escape(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('(', "\\(")
        .replace(')', "\\)")
}

fn heatmap_colour(value: f64, min: f64, max: f64) -> (f64, f64, f64) {
    if !value.is_finite() {
        return (0.85, 0.85, 0.85);
    }
    let t = ((value - min) / (max - min)).clamp(0.0, 1.0);
    if t < 0.5 {
        let q = t * 2.0;
        (0.10 + 0.90 * q, 0.30 + 0.70 * q, 0.80 + 0.20 * q)
    } else {
        let q = (t - 0.5) * 2.0;
        (1.0 - 0.20 * q, 1.0 - 0.90 * q, 1.0 - 0.90 * q)
    }
}

fn write_expression_pdf(
    out: &Path,
    clusters: &Array2<bool>,
    matrices: &[Array2<f64>],
    gdm: &Array2<bool>,
    datasets: &[Dataset],
) -> Result<()> {
    let path = out.join("Cluster_Expression_Profiles.pdf");
    let mut objects: Vec<Vec<u8>> = Vec::new();
    let mut pages = Vec::new();
    let width = 792.0f64;
    let height = 612.0f64;
    let columns = 2usize;
    let panel_width = (width - 60.0) / columns as f64;
    let cluster_count = clusters.ncols();
    let page_count = cluster_count.clamp(1, 3);

    for page in 0..page_count {
        let base = cluster_count / page_count;
        let remainder = cluster_count % page_count;
        let start = page * base + page.min(remainder);
        let page_cluster_count = base + usize::from(page < remainder);
        let end = start + page_cluster_count;
        let page_clusters = end.saturating_sub(start).max(1);
        let rows = (page_clusters + columns - 1) / columns;
        let panel_height = 520.0 / rows as f64;
        let mut content = String::new();
        content.push_str("q 1 1 1 rg 0 0 792 612 re f Q\n");
        content.push_str("0 0 0 rg\n");
        content.push_str(&format!(
            "BT /F1 16 Tf 30 585 Td (Cluster Expression Profiles - page {} of {}) Tj ET\n",
            page + 1,
            page_count
        ));

        for cluster in start..end {
            let members: Vec<usize> = (0..clusters.nrows())
                .filter(|&i| clusters[[i, cluster]])
                .collect();
            let column = (cluster - start) % columns;
            let row = (cluster - start) / columns;
            let panel_x = 30.0 + column as f64 * panel_width;
            let panel_y = 38.0 + (rows - row - 1) as f64 * panel_height;
            content.push_str("0 0 0 rg\n");
            content.push_str(&format!(
                "BT /F1 10 Tf {} {} Td (Cluster {}: {} genes) Tj ET\n",
                panel_x,
                panel_y + panel_height - 30.0,
                cluster + 1,
                members.len()
            ));
            let panel_count = datasets.len().max(1);
            let dataset_panel_height = (panel_height - 34.0) / panel_count as f64;

            for (dataset_idx, dataset) in datasets.iter().enumerate() {
            let x0 = panel_x;
            let y0 = panel_y + (panel_count - dataset_idx - 1) as f64 * dataset_panel_height;
            let plot_w = panel_width * 0.55;
            let heatmap_x = panel_x + panel_width * 0.60;
            let heatmap_w = panel_width * 0.34;
            let plot_h = (dataset_panel_height - 48.0).max(12.0);
            let matrix = &matrices[dataset_idx];
            let present: Vec<usize> = members
                .iter()
                .copied()
                .filter(|&i| gdm[[i, dataset_idx]])
                .collect();
            if present.is_empty() || matrix.ncols() == 0 {
                continue;
            }

            let values: Vec<f64> = present
                .iter()
                .flat_map(|&i| matrix.row(i).to_vec())
                .collect();
            let mut min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let mut max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !min.is_finite() || !max.is_finite() {
                continue;
            }
            if (max - min).abs() < 1e-12 {
                min -= 1.0;
                max += 1.0;
            }
            let pad = (max - min) * 0.05;
            min -= pad;
            max += pad;

            content.push_str("0 0 0 rg\n");
            content.push_str(&format!(
                "BT /F1 7 Tf {} {} Td ({}) Tj ET\n",
                x0,
                y0 + plot_h + 8.0,
                pdf_escape(&dataset.name)
            ));
            content.push_str("0.82 0.82 0.82 RG 0.5 w\n");
            content.push_str(&format!(
                "{} {} m {} {} l S\n{} {} m {} {} l S\n",
                x0,
                y0,
                x0 + plot_w,
                y0,
                x0,
                y0,
                x0,
                y0 + plot_h
            ));
            for &gene in &present {
                content.push_str("0.72 0.72 0.72 RG 0.45 w\n");
                for sample in 0..matrix.ncols() {
                    let value = matrix[[gene, sample]];
                    if !value.is_finite() {
                        continue;
                    }
                    let x = x0
                        + sample as f64 * plot_w / (matrix.ncols().saturating_sub(1).max(1) as f64);
                    let y = y0 + (value - min) / (max - min) * plot_h;
                    content.push_str(&format!(
                        "{} {} {}\n",
                        if sample == 0 {
                            format!("{} {} m", x, y)
                        } else {
                            format!("{} {} l", x, y)
                        },
                        "",
                        ""
                    ));
                }
                content.push_str("S\n");
            }
            content.push_str("0.08 0.25 0.55 RG 1.8 w\n");
            for sample in 0..matrix.ncols() {
                let mean = present.iter().map(|&i| matrix[[i, sample]]).sum::<f64>()
                    / present.len() as f64;
                let x =
                    x0 + sample as f64 * plot_w / (matrix.ncols().saturating_sub(1).max(1) as f64);
                let y = y0 + (mean - min) / (max - min) * plot_h;
                content.push_str(&format!(
                    "{} {} {}\n",
                    if sample == 0 {
                        format!("{} {} m", x, y)
                    } else {
                        format!("{} {} l", x, y)
                    },
                    "",
                    ""
                ));
            }
            content.push_str("S\n");

            content.push_str("0 0 0 rg\n");
            content.push_str(&format!(
                "BT /F1 7 Tf {} {} Td (Heatmap) Tj ET\n",
                heatmap_x,
                y0 + plot_h + 8.0
            ));
            let cell_w = heatmap_w / matrix.ncols() as f64;
            let cell_h = plot_h / present.len() as f64;
            for (row, &gene) in present.iter().enumerate() {
                let cell_y = y0 + (present.len() - row - 1) as f64 * cell_h;
                for sample in 0..matrix.ncols() {
                    let (red, green, blue) = heatmap_colour(matrix[[gene, sample]], min, max);
                    let cell_x = heatmap_x + sample as f64 * cell_w;
                    content.push_str(&format!(
                        "{} {} {} rg {} {} {} {} re f\n",
                        red,
                        green,
                        blue,
                        cell_x,
                        cell_y,
                        cell_w + 0.15,
                        cell_h + 0.15
                    ));
                }
            }
            content.push_str(&format!(
                "0.35 0.35 0.35 RG 0.5 w {} {} {} {} re S\n",
                heatmap_x, y0, heatmap_w, plot_h
            ));
            let x_denominator = matrix.ncols().saturating_sub(1).max(1) as f64;
            for (sample, sample_id) in dataset.sample_ids.iter().take(matrix.ncols()).enumerate() {
                let x = x0 + sample as f64 * plot_w / x_denominator;
                let label_width = sample_id.chars().count() as f64 * 4.0;
                content.push_str(&format!(
                    "0.82 0.82 0.82 RG 0.5 w {} {} m {} {} l S\n",
                    x,
                    y0,
                    x,
                    y0 - 4.0
                ));
                content.push_str("0 0 0 rg\n");
                content.push_str(&format!(
                    "BT /F1 5 Tf {} {} Td ({}) Tj ET\n",
                    x - label_width / 2.0,
                    y0 - 14.0,
                    pdf_escape(sample_id)
                ));
                let heatmap_label_x = heatmap_x + (sample as f64 + 0.5) * cell_w;
                content.push_str(&format!(
                    "0.35 0.35 0.35 RG 0.5 w {} {} m {} {} l S\n",
                    heatmap_label_x,
                    y0,
                    heatmap_label_x,
                    y0 - 4.0
                ));
                content.push_str("0 0 0 rg\n");
                content.push_str(&format!(
                    "BT /F1 5 Tf {} {} Td ({}) Tj ET\n",
                    heatmap_label_x - label_width / 2.0,
                    y0 - 14.0,
                    pdf_escape(sample_id)
                ));
            }
            }
        }

        let stream = content.into_bytes();
        let content_id = objects.len() + 1;
        objects.push(stream);
        pages.push(content_id);
    }

    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");
    let catalog_id = objects.len() + 1;
    let pages_id = catalog_id + 1;
    let font_id = pages_id + 1;
    let mut offsets = vec![0usize];
    for (idx, object) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(
            format!(
                "{} 0 obj\n<< /Length {} >>\nstream\n",
                idx + 1,
                object.len()
            )
            .as_bytes(),
        );
        pdf.extend_from_slice(object);
        pdf.extend_from_slice(b"\nendstream\nendobj\n");
    }
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        format!(
            "{} 0 obj\n<< /Type /Catalog /Pages {} 0 R >>\nendobj\n",
            catalog_id, pages_id
        )
        .as_bytes(),
    );
    offsets.push(pdf.len());
    let kids: String = pages.iter().map(|id| format!("<< /Type /Page /Parent {} 0 R /MediaBox [0 0 {} {}] /Resources << /Font << /F1 {} 0 R >> >> /Contents {} 0 R >>", pages_id, width, height, font_id, id)).collect::<Vec<_>>().join(" ");
    pdf.extend_from_slice(
        format!(
            "{} 0 obj\n<< /Type /Pages /Kids [{}] /Count {} >>\nendobj\n",
            pages_id,
            kids,
            pages.len()
        )
        .as_bytes(),
    );
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        format!(
            "{} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
            font_id
        )
        .as_bytes(),
    );
    let xref = pdf.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len()).as_bytes());
    for offset in offsets.iter().skip(1) {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root {} 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            catalog_id,
            xref
        )
        .as_bytes(),
    );
    fs::write(path, pdf)?;
    Ok(())
}
