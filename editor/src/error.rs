use std::fmt;

#[derive(Debug, PartialEq)]
pub enum Error {
    InvalidBufferPosition(usize),
    InvalidRange(std::ops::Range<usize>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidBufferPosition(ref pos) => write!(f, "invalid buffer position: {}", pos),
            Error::InvalidRange(ref range) => {
                write!(f, "invalid range: start={} end={}", range.start, range.end)
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            _ => None,
        }
    }
}
