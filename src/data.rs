use crate::error::{ClustError, Result};
use ndarray::Array2;
use std::collections::HashMap;
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

/// Replace replicate columns with their arithmetic-mean expression profile.
///
/// The replicate file contains two columns, `sample_id` and `replicate_id`,
/// and may be comma-, tab-, or semicolon-delimited. A header row is optional.
pub fn aggregate_replicates(dataset: &Dataset, path: &Path) -> Result<Dataset> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();
    let first = lines
        .next()
        .ok_or_else(|| ClustError::InvalidData("Replicate file is empty".into()))??;
    let delim = detect_delimiter(&first);
    let mut mappings = Vec::new();
    let mut seen_samples = HashMap::new();

    let mut parse_line = |line: &str, line_no: usize| -> Result<()> {
        let fields: Vec<&str> = line.split(delim).map(str::trim).collect();
        if fields.len() != 2 || fields.iter().any(|field| field.is_empty()) {
            return Err(ClustError::Parse(format!(
                "replicate file line {line_no}: expected sample_id and replicate_id"
            )));
        }
        if seen_samples
            .insert(fields[0].to_string(), line_no)
            .is_some()
        {
            return Err(ClustError::InvalidData(format!(
                "sample '{}' occurs more than once in replicate file",
                fields[0]
            )));
        }
        mappings.push((fields[0].to_string(), fields[1].to_string()));
        Ok(())
    };

    let first_fields: Vec<&str> = first.split(delim).map(str::trim).collect();
    if first_fields.len() == 2 && dataset.sample_ids.iter().any(|id| id == first_fields[0]) {
        parse_line(&first, 1)?;
    }
    for (line_no, line) in lines.enumerate() {
        let line = line?;
        if !line.trim().is_empty() {
            parse_line(&line, line_no + 2)?;
        }
    }
    if mappings.is_empty() {
        return Err(ClustError::InvalidData(
            "Replicate file contains no sample mappings".into(),
        ));
    }

    let mut sample_to_column = HashMap::new();
    for (index, sample) in dataset.sample_ids.iter().enumerate() {
        sample_to_column.insert(sample.as_str(), index);
    }
    if let Some((sample, _)) = mappings
        .iter()
        .find(|(sample, _)| !sample_to_column.contains_key(sample.as_str()))
    {
        return Err(ClustError::InvalidData(format!(
            "Replicate file references unknown sample '{sample}'"
        )));
    }
    if let Some(sample) = dataset
        .sample_ids
        .iter()
        .find(|sample| !seen_samples.contains_key(sample.as_str()))
    {
        return Err(ClustError::InvalidData(format!(
            "No replicate assignment for sample '{sample}'"
        )));
    }

    let mut replicate_ids = Vec::new();
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (sample, replicate) in mappings {
        if !groups.contains_key(&replicate) {
            replicate_ids.push(replicate.clone());
        }
        groups
            .entry(replicate)
            .or_default()
            .push(sample_to_column[sample.as_str()]);
    }

    let mut matrix = Array2::<f64>::zeros((dataset.n_genes(), replicate_ids.len()));
    for (replicate_index, replicate) in replicate_ids.iter().enumerate() {
        let columns = &groups[replicate];
        for gene in 0..dataset.n_genes() {
            let sum: f64 = columns
                .iter()
                .map(|&column| dataset.matrix[[gene, column]])
                .sum();
            matrix[[gene, replicate_index]] = sum / columns.len() as f64;
        }
    }
    Ok(Dataset {
        name: dataset.name.clone(),
        gene_ids: dataset.gene_ids.clone(),
        sample_ids: replicate_ids,
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

#[cfg(test)]
mod tests {
    use super::{aggregate_replicates, Dataset};
    use ndarray::array;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn mapping_path(contents: &str) -> PathBuf {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("clust-rs-replicates-{suffix}-{id}.tsv"));
        fs::write(&path, contents).expect("write mapping");
        path
    }

    #[test]
    fn aggregates_samples_in_first_seen_replicate_order() {
        let dataset = Dataset {
            name: "test".into(),
            gene_ids: vec!["g1".into(), "g2".into()],
            sample_ids: vec!["s1".into(), "s2".into(), "s3".into()],
            matrix: array![[1.0, 3.0, 10.0], [2.0, 4.0, 20.0]],
        };
        let path = mapping_path("sample_id\treplicate_id\ns1\tgroup_a\ns2\tgroup_a\ns3\tgroup_b\n");
        let result = aggregate_replicates(&dataset, &path).expect("valid mapping");
        fs::remove_file(path).expect("remove mapping");

        assert_eq!(result.sample_ids, ["group_a", "group_b"]);
        assert_eq!(result.matrix, array![[2.0, 10.0], [3.0, 20.0]]);
    }

    #[test]
    fn rejects_unassigned_samples() {
        let dataset = Dataset {
            name: "test".into(),
            gene_ids: vec!["g1".into()],
            sample_ids: vec!["s1".into(), "s2".into()],
            matrix: array![[1.0, 2.0]],
        };
        let path = mapping_path("s1\tgroup_a\n");
        let result = aggregate_replicates(&dataset, &path);
        fs::remove_file(path).expect("remove mapping");

        assert!(result
            .expect_err("incomplete mapping should fail")
            .to_string()
            .contains("No replicate assignment for sample 's2'"));
    }
}
