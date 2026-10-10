use thiserror::Error;

#[derive(Error, Debug)]
pub enum AirError {
    #[error("I/O Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("GGUF Parsing Error: {0}")]
    GgufParse(String),

    #[error("Invalid Tensor Shape: expected {expected:?}, got {actual:?}")]
    InvalidShape { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Quantization Error: {0}")]
    Quantization(String),

    #[error("Tokenizer Error: {0}")]
    Tokenizer(String),

    #[error("Hardware Abstraction Error: {0}")]
    Hal(String),

    #[error("Context Engine Error: {0}")]
    Ice(String),

    #[error("Execution Engine Error: {0}")]
    Engine(String),

    #[error("Multimodal Ingestion Error: {0}")]
    Mie(String),

    #[error("Server Error: {0}")]
    Server(String),

    #[error("Model Download Error: {0}")]
    Hub(String),

    #[error("Config Error: {0}")]
    Config(String),

    #[error("Generic Error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, AirError>;
