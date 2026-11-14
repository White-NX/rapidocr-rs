//! Public, machine-readable errors shared by all synchronous OCR APIs.
//!
//! Match [`OcrError`] or use [`OcrError::kind`] to choose recovery without
//! inspecting diagnostic text. Backend and I/O causes remain available through
//! [`std::error::Error::source`]. Tokio wraps these errors in
//! `TokioOcrError::Ocr` while keeping queue, timeout, and worker states explicit.

use std::{
    error::Error as StdError,
    path::{Path, PathBuf},
};

use thiserror::Error;

use crate::cancellation::OcrCancelled;

/// Result returned by the OCR, configuration, model-cache, and runtime APIs.
pub type OcrResult<T> = std::result::Result<T, OcrError>;

/// Original backend or I/O cause, retained without converting it to text.
pub type ErrorSource = Box<dyn StdError + Send + Sync + 'static>;

/// Broad recovery categories for OCR failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OcrErrorKind {
    /// The request was cooperatively cancelled.
    Cancelled,
    /// Repair the configuration before trying again.
    Configuration,
    /// Repair, replace, or prepare the model or dictionary.
    Model,
    /// Inspect the runtime or execution provider and the original cause.
    Runtime,
    /// Repair the input image or its path before trying again.
    InvalidInput,
    /// Tile the input, reduce batching, or explicitly adjust resource limits.
    InputLimit,
}

/// Configuration failure categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigErrorKind {
    /// Invalid TOML syntax or a value with the wrong type.
    Parse,
    /// A field is not part of the configuration schema.
    UnknownField,
    /// Legacy and nested pipeline flags disagree.
    ConflictingField,
    /// A field range or pipeline combination is invalid.
    Validation,
    /// Reading or writing a configuration file failed.
    Io,
    /// Serializing a configuration failed.
    Serialization,
    /// A path cannot be resolved against the requested base.
    PathResolution,
}

/// Model and dictionary failure categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelErrorKind {
    /// A required asset does not exist.
    NotFound,
    /// A model, dictionary, or cache cannot be read or written.
    Io,
    /// An asset failed checksum verification.
    Integrity,
    /// The model is invalid or disagrees with the configured tensor contract.
    Incompatible,
    /// Fetching a model failed.
    Download,
    /// Fetching a missing model requires the disabled model-download feature.
    DownloadDisabled,
}

/// Runtime failure categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuntimeErrorKind {
    /// Loading a library or initializing an environment failed.
    Initialization,
    /// Configuring a session or execution provider failed.
    SessionConfiguration,
    /// Running inference failed.
    Execution,
    /// Creating run options or signaling termination failed.
    Cancellation,
    /// Creating the dedicated Tokio worker failed.
    WorkerSpawn,
    /// The dedicated Tokio worker panicked.
    WorkerPanic,
}

/// Stage whose input exceeded a resource limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OcrStage {
    /// High-level image resizing or padding.
    Preprocessing,
    /// Text detection.
    Detection,
    /// Text-line direction classification.
    Classification,
    /// Text recognition.
    Recognition,
}

/// Resource that prevented accepting an input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InputLimitKind {
    /// Maximum detector side length.
    SideLength,
    /// Maximum detector pixel count.
    PixelCount,
    /// Maximum padded recognition width.
    Width,
    /// Maximum padded pixels in a recognition batch.
    BatchPixelCount,
    /// Image or tensor dimensions overflow the platform's representation.
    Dimensions,
    /// Decoding an image exceeds the decoder's allocation limit.
    ImageBytes,
    /// Tensor bytes exceed the platform's allocation limit.
    TensorBytes,
}

/// Categorized OCR failure with its original cause and recovery metadata.
///
/// Diagnostic messages are intended for humans and may change. Variants,
/// category enums, paths, stages, and numeric limits are the matching API.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OcrError {
    /// The operation was cooperatively cancelled.
    #[error(transparent)]
    Cancelled(#[from] OcrCancelled),
    /// The configuration could not be read, parsed, validated, or resolved.
    #[error("{message}")]
    #[non_exhaustive]
    Config {
        /// Configuration failure category.
        kind: ConfigErrorKind,
        /// Dotted field path, when a specific field is known.
        field: Option<String>,
        /// Human-readable explanation.
        message: String,
        /// Original parser or I/O error.
        #[source]
        source: Option<ErrorSource>,
    },
    /// A model, dictionary, or model cache failed preparation or validation.
    #[error("{message}")]
    #[non_exhaustive]
    Model {
        /// Model failure category.
        kind: ModelErrorKind,
        /// Related asset or cache path, when known.
        path: Option<PathBuf>,
        /// Human-readable explanation.
        message: String,
        /// Original ONNX, network, or I/O error.
        #[source]
        source: Option<ErrorSource>,
    },
    /// Runtime initialization, setup, or execution failed.
    #[error("{message}")]
    #[non_exhaustive]
    Runtime {
        /// Runtime failure category.
        kind: RuntimeErrorKind,
        /// Human-readable explanation.
        message: String,
        /// Original ONNX or I/O error.
        #[source]
        source: Option<ErrorSource>,
    },
    /// The input image or path is invalid.
    #[error("{message}")]
    #[non_exhaustive]
    InvalidInput {
        /// Related image path, if supplied by the caller.
        path: Option<PathBuf>,
        /// Human-readable explanation.
        message: String,
        /// Original image-decoder or I/O error.
        #[source]
        source: Option<ErrorSource>,
    },
    /// An input exceeds a configured or platform resource limit.
    #[error("{message}")]
    #[non_exhaustive]
    InputLimit {
        /// Stage requiring the oversized input.
        stage: OcrStage,
        /// Resource that exceeded its limit.
        kind: InputLimitKind,
        /// Requested amount, or `None` when unavailable (for example, arithmetic
        /// overflow or a decoder that does not expose its allocation request).
        requested: Option<u64>,
        /// Applicable limit, or `None` when the backend does not expose it.
        limit: Option<u64>,
        /// Human-readable explanation.
        message: String,
        /// Original cause, if reported by an image decoder.
        #[source]
        source: Option<ErrorSource>,
    },
}

impl OcrError {
    /// Returns the broad recovery category without inspecting error text.
    pub const fn kind(&self) -> OcrErrorKind {
        match self {
            Self::Cancelled(_) => OcrErrorKind::Cancelled,
            Self::Config { .. } => OcrErrorKind::Configuration,
            Self::Model { .. } => OcrErrorKind::Model,
            Self::Runtime { .. } => OcrErrorKind::Runtime,
            Self::InvalidInput { .. } => OcrErrorKind::InvalidInput,
            Self::InputLimit { .. } => OcrErrorKind::InputLimit,
        }
    }

    pub(crate) fn config(
        kind: ConfigErrorKind,
        field: Option<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::Config {
            kind,
            field,
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn invalid_config(message: impl Into<String>) -> Self {
        Self::config(ConfigErrorKind::Validation, None, message)
    }

    pub(crate) fn model(
        kind: ModelErrorKind,
        path: Option<&Path>,
        message: impl Into<String>,
    ) -> Self {
        Self::Model {
            kind,
            path: path.map(Path::to_path_buf),
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn incompatible_model(message: impl Into<String>) -> Self {
        Self::model(ModelErrorKind::Incompatible, None, message)
    }

    pub(crate) fn runtime(kind: RuntimeErrorKind, message: impl Into<String>) -> Self {
        Self::Runtime {
            kind,
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn invalid_input(path: Option<&Path>, message: impl Into<String>) -> Self {
        Self::InvalidInput {
            path: path.map(Path::to_path_buf),
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn input_limit(
        stage: OcrStage,
        kind: InputLimitKind,
        requested: Option<u64>,
        limit: Option<u64>,
        message: impl Into<String>,
    ) -> Self {
        Self::InputLimit {
            stage,
            kind,
            requested,
            limit,
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn with_source(mut self, cause: impl StdError + Send + Sync + 'static) -> Self {
        match &mut self {
            Self::Config { source, .. }
            | Self::Model { source, .. }
            | Self::Runtime { source, .. }
            | Self::InvalidInput { source, .. }
            | Self::InputLimit { source, .. } => *source = Some(Box::new(cause)),
            Self::Cancelled(_) => {}
        }
        self
    }

    pub(crate) fn with_model_path(mut self, model_path: &Path) -> Self {
        if let Self::Model { path, .. } = &mut self {
            if path.is_none() {
                *path = Some(model_path.to_path_buf());
            }
        }
        self
    }

    fn with_context(mut self, context: impl AsRef<str>) -> Self {
        match &mut self {
            Self::Config { message, .. }
            | Self::Model { message, .. }
            | Self::Runtime { message, .. }
            | Self::InvalidInput { message, .. }
            | Self::InputLimit { message, .. } => {
                *message = format!("{}: {message}", context.as_ref())
            }
            Self::Cancelled(_) => {}
        }
        self
    }
}

// Context keeps the error variant and cause intact so public callers always
// match the original category, even after stage-specific diagnostics are added.
pub(crate) trait ResultExt<T> {
    fn context(self, context: impl AsRef<str>) -> OcrResult<T>;
    fn with_context(self, context: impl FnOnce() -> String) -> OcrResult<T>;
}

impl<T> ResultExt<T> for OcrResult<T> {
    fn context(self, context: impl AsRef<str>) -> OcrResult<T> {
        self.map_err(|error| error.with_context(context))
    }

    fn with_context(self, context: impl FnOnce() -> String) -> OcrResult<T> {
        self.map_err(|error| error.with_context(context()))
    }
}

macro_rules! ensure_config {
    ($condition:expr, $($message:tt)*) => {
        if !$condition { return Err($crate::OcrError::invalid_config(format!($($message)*))); }
    };
}

macro_rules! ensure_model {
    ($condition:expr, $($message:tt)*) => {
        if !$condition { return Err($crate::OcrError::incompatible_model(format!($($message)*))); }
    };
}

pub(crate) use {ensure_config, ensure_model};
