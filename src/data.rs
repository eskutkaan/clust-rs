use crate::error::{ClustError, Result};
use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Dataset {
    pub name: String,
    pub gene_ids: Vec<String>,
    pub sample_ids: Vec<String>,
    pub matrix: Array2<f64>,
}

impl Dataset {
    pub fn n_genes(&self) -> usize {
        self.matrix.nrows()
    }

    pub fn n_samples(&self) -> usize {
        self.matrix.ncols()
    }
}

fn detect_delimiter(line: &str) -> char {
    let tab = line.matches('\t').count();
    let comma = line.matches(',').count();
    let semi = line.matches(';').count();
    if tab >= comma && tab >= semi {
        '\t'
    } else if comma >= semi {
        ','
    } else {
        ';'
    }
}

pub fn load_dataset(path: &Path) -> Result<Dataset> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().ok_or(ClustError::EmptyDataset)??;
    let delim = detect_delimiter(&header);
    let sample_ids: Vec<String> = header
        .split(delim)
        .skip(1)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if sample_ids.is_empty() {
        return Err(ClustError::InvalidData("No sample columns".into()));
    }

    let n_samples = sample_ids.len();
    let mut gene_ids = Vec::new();
    let mut rows = Vec::new();
    for (line_no, line) in lines.enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split(delim);
        let gene = parts
            .next()
            .ok_or_else(|| ClustError::Parse(format!("line {}", line_no + 2)))?
            .trim()
            .to_string();
        if gene.is_empty() {
            continue;
        }
        let mut values = Vec::with_capacity(n_samples);
        for value in parts.take(n_samples) {
            let value = value
                .trim()
                .parse()
                .map_err(|_| ClustError::Parse(format!("bad value gene {gene}")))?;
            values.push(value);
        }
        if values.len() != n_samples {
            return Err(ClustError::Parse(format!(
                "gene {gene}: expected {n_samples} values"
            )));
        }
        gene_ids.push(gene);
        rows.push(values);
    }
    if gene_ids.is_empty() {
        return Err(ClustError::EmptyDataset);
    }

    let mut matrix = Array2::<f64>::zeros((gene_ids.len(), n_samples));
    for (i, row) in rows.into_iter().enumerate() {
        for (j, value) in row.into_iter().enumerate() {
            matrix[[i, j]] = value;
        }
    }
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset")
        .to_string();
    Ok(Dataset {
        name,
        gene_ids,
        sample_ids,
        matrix,
    })
}

pub fn subset_present(mat: &Array2<f64>, present: &[bool]) -> (Array2<f64>, Vec<usize>) {
    let indices: Vec<usize> = present
        .iter()
        .enumerate()
        .filter_map(|(i, &is_present)| is_present.then_some(i))
        .collect();
    if indices.is_empty() {
        return (Array2::zeros((0, mat.ncols())), indices);
    }
    (mat.select(ndarray::Axis(0), &indices), indices)
}
