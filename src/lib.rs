//! Clust-RS consensus clustering library.
pub mod cluster;
pub mod data;
pub mod error;
pub mod normalise;
pub mod pipeline;
pub mod select;
pub mod uncles;

pub use error::{ClustError, Result};
pub use pipeline::{run, run_from_matrices, PipelineConfig, PipelineResult};
