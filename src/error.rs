use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClustError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("Empty dataset")]
    EmptyDataset,
}

pub type Result<T> = std::result::Result<T, ClustError>;
