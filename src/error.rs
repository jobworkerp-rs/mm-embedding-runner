use thiserror::Error;

// The Display prefix of each variant ("InvalidArgument:", "NotLoaded:",
// etc.) is the public error classification surfaced to callers. Tests and
// callers match on it via starts_with, so changing these strings is a
// breaking change.
#[derive(Error, Debug)]
pub enum MmEmbeddingError {
    #[error("InvalidArgument: {0}")]
    InvalidArgument(String),

    #[error("NotLoaded: model is not loaded")]
    NotLoaded,

    #[error("BackendError: {0}")]
    BackendError(String),

    #[error("IoError: {0}")]
    IoError(String),

    #[error("UnsupportedDevice: {0}")]
    UnsupportedDevice(String),
}

pub type Result<T> = std::result::Result<T, MmEmbeddingError>;

impl MmEmbeddingError {
    pub fn invalid_argument<S: Into<String>>(msg: S) -> Self {
        Self::InvalidArgument(msg.into())
    }

    pub fn backend<S: Into<String>>(msg: S) -> Self {
        Self::BackendError(msg.into())
    }

    pub fn io<S: Into<String>>(msg: S) -> Self {
        Self::IoError(msg.into())
    }

    pub fn unsupported_device<S: Into<String>>(msg: S) -> Self {
        Self::UnsupportedDevice(msg.into())
    }
}

impl From<std::io::Error> for MmEmbeddingError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
}

impl From<prost::DecodeError> for MmEmbeddingError {
    fn from(err: prost::DecodeError) -> Self {
        // Decode failures originate from caller-supplied bytes, so surface
        // them as InvalidArgument rather than a backend fault.
        Self::InvalidArgument(format!("failed to decode protobuf: {err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_argument_prefix() {
        let err = MmEmbeddingError::invalid_argument("missing field");
        assert!(err.to_string().starts_with("InvalidArgument:"));
    }

    #[test]
    fn not_loaded_prefix() {
        let err = MmEmbeddingError::NotLoaded;
        assert!(err.to_string().starts_with("NotLoaded:"));
    }

    #[test]
    fn backend_prefix() {
        let err = MmEmbeddingError::backend("fastembed failure");
        assert!(err.to_string().starts_with("BackendError:"));
    }

    #[test]
    fn io_prefix() {
        let err = MmEmbeddingError::io("read failed");
        assert!(err.to_string().starts_with("IoError:"));
    }

    #[test]
    fn unsupported_device_prefix() {
        let err = MmEmbeddingError::unsupported_device("cuda feature disabled");
        assert!(err.to_string().starts_with("UnsupportedDevice:"));
    }

    #[test]
    fn from_decode_error_yields_invalid_argument() {
        // Feed prost an obviously malformed byte stream to obtain a real
        // DecodeError without depending on the deprecated public new() ctor.
        // 0xff is a varint with the continuation bit set but no following
        // byte, which is an unambiguous decode failure.
        use prost::Message;
        let result = crate::protobuf::mm_embedding::EmbedTextArgs::decode(&[0xff][..]);
        let decode_err = result.expect_err("malformed varint should fail");
        let mm: MmEmbeddingError = decode_err.into();
        assert!(mm.to_string().starts_with("InvalidArgument:"));
    }

    #[test]
    fn from_io_error_yields_io_error() {
        let io = std::io::Error::other("disk gone");
        let mm: MmEmbeddingError = io.into();
        assert!(mm.to_string().starts_with("IoError:"));
    }
}
