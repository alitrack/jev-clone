//! Content hashing for the frozen item set.
//!
//! SHA-256 over the raw bytes of the file (not over a re-serialisation), so the
//! hash pins exactly what is on disk — including whitespace and key order, which
//! `serde_json::Value` would normalise away.

use sha2::{Digest, Sha256};

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        // `write!` into a String cannot fail; the expect documents that.
        write!(out, "{byte:02x}").expect("writing to a String never fails");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::sha256_hex;

    #[test]
    fn matches_the_published_fips_180_4_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn is_stable_across_calls_and_lengths() {
        // 100 bytes: crosses the SHA-256 block boundary from the padding side.
        let long = "a".repeat(100);
        assert_eq!(sha256_hex(long.as_bytes()), sha256_hex(long.as_bytes()));
        assert_ne!(sha256_hex(long.as_bytes()), sha256_hex(b"a"));
        assert_eq!(sha256_hex(b"abc").len(), 64);
    }
}
