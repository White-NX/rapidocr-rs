//! ONNX Runtime initialization and sharing.
//!
//! [`OcrRuntime`] separates process-wide runtime setup from model loading. The
//! default runtime is initialized once and reused by every OCR pipeline. A
//! caller managing other ONNX models can instead supply its own environment.

use std::sync::Mutex;

#[cfg(feature = "load-dynamic")]
use std::path::{Path, PathBuf};

use ort::environment::Environment;

use crate::error::{OcrError, OcrResult as Result, RuntimeErrorKind};

// ORT permits only one environment per process, including after the last
// session is dropped. Keep a strong handle so later pipelines can reuse it.
static DEFAULT_RUNTIME: Mutex<Option<OcrRuntime>> = Mutex::new(None);

/// Shared ONNX Runtime environment used to construct OCR sessions.
///
/// Cloning this handle shares the environment; it does not duplicate models or
/// sessions. Session thread counts and execution providers remain configured
/// through [`InferenceOptions`](crate::config::InferenceOptions).
#[derive(Debug, Clone)]
pub struct OcrRuntime {
    environment: Environment,
    #[cfg(feature = "load-dynamic")]
    library_path: Option<PathBuf>,
}

impl OcrRuntime {
    /// Returns the process-wide default runtime, initializing it if necessary.
    ///
    /// With `load-dynamic`, loads `ORT_DYLIB_PATH` or the platform's default
    /// `onnxruntime` library name and reports loading failures as errors. Call
    /// `OcrRuntime::load_from` first to select an explicit library path.
    ///
    /// If the application has already created an `ort` environment itself,
    /// wrap that environment with [`OcrRuntime::from_environment`] and pass the
    /// resulting handle to [`RapidOcr::new_with_runtime`](crate::RapidOcr::new_with_runtime).
    pub fn new() -> Result<Self> {
        let mut cached = DEFAULT_RUNTIME
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(runtime) = cached.as_ref() {
            return Ok(runtime.clone());
        }

        #[cfg(feature = "load-dynamic")]
        let path = std::env::var_os("ORT_DYLIB_PATH")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(format!(
                    "{}onnxruntime{}",
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                ))
            });
        #[cfg(feature = "load-dynamic")]
        let builder = ort::init_from(&path).map_err(|source| {
            OcrError::runtime(
                RuntimeErrorKind::Initialization,
                format!("failed to load ONNX Runtime from {}", path.display()),
            )
            .with_source(source)
        })?;
        #[cfg(not(feature = "load-dynamic"))]
        let builder = ort::init();

        let runtime = Self {
            environment: builder
                .with_name("rapidocr")
                .with_telemetry(false)
                .build()
                .map_err(|source| OcrError::runtime(RuntimeErrorKind::Initialization, "failed to initialize ONNX Runtime; if the application already owns an ort environment, use OcrRuntime::from_environment and new_with_runtime").with_source(source))?,
            #[cfg(feature = "load-dynamic")]
            library_path: Some(resolve_library_path(&path)),
        };
        *cached = Some(runtime.clone());
        Ok(runtime)
    }

    /// Loads an application-provided ONNX Runtime shared library.
    ///
    /// Requires `load-dynamic`. The library must implement ONNX Runtime API
    /// 1.24 or newer. This call must precede any other `ort` use in the process.
    /// Repeated calls for the same path share the runtime; a different path
    /// after initialization returns an error rather than silently changing it.
    #[cfg(feature = "load-dynamic")]
    pub fn load_from(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let path = std::fs::canonicalize(path).map_err(|source| {
            OcrError::runtime(
                RuntimeErrorKind::Initialization,
                format!("ONNX Runtime library not found at {}", path.display()),
            )
            .with_source(source)
        })?;
        if !path.is_file() {
            return Err(OcrError::runtime(
                RuntimeErrorKind::Initialization,
                format!(
                    "ONNX Runtime library path is not a file: {}",
                    path.display()
                ),
            ));
        }

        let mut cached = DEFAULT_RUNTIME
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(runtime) = cached.as_ref() {
            if runtime.library_path.as_ref() != Some(&path) {
                return Err(OcrError::runtime(RuntimeErrorKind::Initialization, "ONNX Runtime is already initialized with another library; select the library before constructing any OCR pipeline"));
            }
            return Ok(runtime.clone());
        }

        let environment = ort::init_from(&path)
            .map_err(|source| {
                OcrError::runtime(
                    RuntimeErrorKind::Initialization,
                    format!("failed to load ONNX Runtime from {}", path.display()),
                )
                .with_source(source)
            })?
            .with_name("rapidocr")
            .with_telemetry(false)
            .build()
            .map_err(|source| {
                OcrError::runtime(
                    RuntimeErrorKind::Initialization,
                    "failed to initialize ONNX Runtime environment",
                )
                .with_source(source)
            })?;
        let runtime = Self {
            environment,
            library_path: Some(path),
        };
        *cached = Some(runtime.clone());
        Ok(runtime)
    }

    /// Wraps an environment already owned by the application.
    ///
    /// Use this explicit interoperability entry point when OCR shares a process
    /// with other ONNX models. The caller controls logging, telemetry, global
    /// thread pools, and library loading. Pass this handle to each OCR pipeline
    /// or Tokio service instead of creating another default environment.
    ///
    /// The application must use the same pinned `ort` version as this crate.
    pub fn from_environment(environment: Environment) -> Self {
        Self {
            environment,
            #[cfg(feature = "load-dynamic")]
            library_path: None,
        }
    }

    /// Borrows the environment for application-owned ONNX sessions.
    ///
    /// This is an explicit `ort` interoperability API and requires the same
    /// pinned `ort` version as this crate.
    pub fn environment(&self) -> &Environment {
        &self.environment
    }
}

#[cfg(feature = "load-dynamic")]
fn resolve_library_path(path: &Path) -> PathBuf {
    if !path.is_absolute() {
        if let Ok(executable) = std::env::current_exe() {
            if let Some(parent) = executable.parent() {
                let candidate = parent.join(path);
                if let Ok(absolute) = std::fs::canonicalize(candidate) {
                    return absolute;
                }
            }
        }
    }
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}
