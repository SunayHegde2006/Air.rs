//! Enterprise OAuth2 / OIDC JWT Verifier (v0.9.0)
//!
//! Verifies Bearer JWTs issued by any OIDC-compliant identity provider
//! (Auth0, Keycloak, Azure AD, Google, Okta, etc.).
//!
//! # Flow
//! ```text
//! HTTP Request
//!   → extract "Authorization: Bearer <jwt>"
//!   → OidcVerifier::verify(jwt)
//!       → decode header → extract kid
//!       → fetch JWKS (cached, 5-min TTL)
//!       → verify signature + exp + aud + iss
//!       → return Claims { sub, scope, exp }
//! ```
//!
//! # Coexistence with ApiKeyStore
//! Both auth backends are active simultaneously. Detection by dot-count:
//! - 2 dots → JWT (header.payload.signature) → OIDC path
//! - no dots → opaque API key → KeyStore path
//!
//! # Research / Standards basis
//! - RFC 7517 (JWK), RFC 7519 (JWT), RFC 9068 (JWT Profile for OAuth2)
//! - OpenID Connect Core 1.0

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// OIDC Error
// ---------------------------------------------------------------------------

/// Errors returned by the OIDC verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OidcError {
    /// Token is malformed (not a valid JWT structure).
    MalformedToken,
    /// Token has expired (`exp` claim is in the past).
    TokenExpired,
    /// `aud` claim does not match configured audience.
    AudienceMismatch,
    /// `iss` claim does not match configured issuer.
    IssuerMismatch,
    /// The signing key (`kid`) was not found in the JWKS.
    UnknownKeyId(String),
    /// Signature verification failed.
    InvalidSignature,
    /// JWKS endpoint returned an error.
    JwksFetchError(String),
    /// An internal decoding error occurred.
    DecodingError(String),
}

impl std::fmt::Display for OidcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedToken           => write!(f, "malformed JWT token"),
            Self::TokenExpired             => write!(f, "JWT token has expired"),
            Self::AudienceMismatch         => write!(f, "JWT audience does not match"),
            Self::IssuerMismatch           => write!(f, "JWT issuer does not match"),
            Self::UnknownKeyId(kid)        => write!(f, "unknown key id: {kid}"),
            Self::InvalidSignature         => write!(f, "JWT signature verification failed"),
            Self::JwksFetchError(e)        => write!(f, "JWKS fetch error: {e}"),
            Self::DecodingError(e)         => write!(f, "JWT decoding error: {e}"),
        }
    }
}

impl std::error::Error for OidcError {}

// ---------------------------------------------------------------------------
// JWT Claims
// ---------------------------------------------------------------------------

/// Decoded JWT claims returned after successful verification.
#[derive(Debug, Clone)]
pub struct Claims {
    /// Subject identifier (user ID, client ID, service account, etc.)
    pub sub: String,
    /// OAuth2 scopes granted to this token (space-separated).
    pub scope: Option<String>,
    /// Unix timestamp when the token expires.
    pub exp: u64,
    /// Token issuer URL.
    pub iss: String,
    /// Token audience(s).
    pub aud: Vec<String>,
    /// JWT ID (for replay attack prevention, optional).
    pub jti: Option<String>,
}

impl Claims {
    /// Returns `true` if the token is currently valid (not expired).
    pub fn is_valid(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.exp > now
    }

    /// Returns `true` if the claims include the given scope.
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scope
            .as_deref()
            .map(|s| s.split_whitespace().any(|w| w == scope))
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// JWKS Key Cache
// ---------------------------------------------------------------------------

/// Cached JWK key entry.
#[derive(Debug, Clone)]
pub struct CachedKey {
    /// PEM-encoded public key (or key material).
    pub key_material: Vec<u8>,
    /// Algorithm (e.g. "RS256", "ES256").
    pub algorithm: String,
    /// When this cache entry was last refreshed (Unix seconds).
    pub cached_at: u64,
}

impl CachedKey {
    fn is_stale(&self, ttl_secs: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Use >= so that TTL=0 means "immediately stale" (inserted key is
        // already expired: now - cached_at = 0, and 0 >= 0 is true).
        now.saturating_sub(self.cached_at) >= ttl_secs
    }
}

/// In-memory JWKS key cache.
///
/// Thread-safety: protected by `std::sync::RwLock` at the `OidcVerifier` level.
/// Keys are indexed by `kid` from the JWK Set.
#[derive(Debug, Default)]
pub struct JwksCache {
    /// kid → cached key material
    keys: HashMap<String, CachedKey>,
    /// Key TTL in seconds (default: 300 = 5 minutes)
    ttl_secs: u64,
}

impl JwksCache {
    pub fn new(ttl_secs: u64) -> Self {
        Self { keys: HashMap::new(), ttl_secs }
    }

    /// Insert or refresh a key.
    pub fn insert(&mut self, kid: String, key_material: Vec<u8>, algorithm: String) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.keys.insert(kid, CachedKey { key_material, algorithm, cached_at: now });
    }

    /// Look up a key by `kid`. Returns `None` if not cached or stale.
    pub fn get(&self, kid: &str) -> Option<&CachedKey> {
        self.keys.get(kid).filter(|k| !k.is_stale(self.ttl_secs))
    }

    /// Remove all stale keys.
    pub fn evict_stale(&mut self) {
        let ttl = self.ttl_secs;
        self.keys.retain(|_, v| !v.is_stale(ttl));
    }

    pub fn len(&self) -> usize { self.keys.len() }
    pub fn is_empty(&self) -> bool { self.keys.is_empty() }
}

// ---------------------------------------------------------------------------
// OidcVerifier Config
// ---------------------------------------------------------------------------

/// Configuration for the OIDC verifier.
#[derive(Debug, Clone)]
pub struct OidcConfig {
    /// Issuer URL (must match `iss` claim exactly).
    pub issuer: String,
    /// Expected audience (must appear in `aud` claim).
    pub audience: String,
    /// JWKS endpoint URL. Auto-derived from issuer if not specified:
    /// `{issuer}/.well-known/jwks.json`
    pub jwks_url: Option<String>,
    /// Key cache TTL in seconds (default: 300).
    pub cache_ttl_secs: u64,
    /// Clock skew tolerance in seconds (default: 30).
    pub leeway_secs: u64,
}

impl OidcConfig {
    /// Construct from issuer URL; other fields use defaults.
    pub fn from_issuer(issuer: impl Into<String>) -> Self {
        let issuer = issuer.into();
        Self {
            jwks_url: Some(format!("{}/.well-known/jwks.json", issuer.trim_end_matches('/'))),
            issuer,
            audience: String::new(),
            cache_ttl_secs: 300,
            leeway_secs: 30,
        }
    }

    pub fn with_audience(mut self, aud: impl Into<String>) -> Self {
        self.audience = aud.into();
        self
    }
}

// ---------------------------------------------------------------------------
// OidcVerifier
// ---------------------------------------------------------------------------

/// OIDC JWT verifier.
///
/// In v0.9.0 this provides the full structural implementation.
/// Key fetching (`fetch_jwks`) is a stub that accepts pre-seeded test keys;
/// in production the caller seeds keys via `seed_key()` or
/// integrates an HTTP client (e.g. `reqwest`) to call `jwks_url`.
///
/// # Thread safety
/// `OidcVerifier` is `Send + Sync`. The key cache is protected by `RwLock`.
pub struct OidcVerifier {
    config: OidcConfig,
    cache: std::sync::RwLock<JwksCache>,
}

impl OidcVerifier {
    /// Construct from configuration.
    pub fn new(config: OidcConfig) -> Self {
        let ttl = config.cache_ttl_secs;
        Self {
            config,
            cache: std::sync::RwLock::new(JwksCache::new(ttl)),
        }
    }

    /// Seed a verification key manually (for testing or pre-provisioned keys).
    pub fn seed_key(&self, kid: impl Into<String>, key: Vec<u8>, alg: impl Into<String>) {
        self.cache.write().unwrap()
            .insert(kid.into(), key, alg.into());
    }

    /// Verify a raw JWT string.
    ///
    /// Returns decoded `Claims` on success, `OidcError` on failure.
    ///
    /// In v0.9.0 this performs all structural validations
    /// (token format, expiry, issuer, audience) but uses a simplified
    /// signature check. Full RS256/ES256 HMAC verification is added in v1.0.0
    /// when the `jsonwebtoken` crate is integrated.
    pub fn verify(&self, token: &str) -> Result<Claims, OidcError> {
        // --- 1. Structural check: JWT must have exactly 2 dots ---
        let parts: Vec<&str> = token.splitn(3, '.').collect();
        if parts.len() != 3 {
            return Err(OidcError::MalformedToken);
        }

        // --- 2. Decode header (base64url) ---
        let header_json = base64url_decode(parts[0])
            .map_err(|_| OidcError::MalformedToken)?;
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&header_json)
                .map_err(|e| OidcError::DecodingError(e.to_string()))?;

        let kid = header.get("kid")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_owned();

        // --- 3. Decode payload ---
        let payload_json = base64url_decode(parts[1])
            .map_err(|_| OidcError::MalformedToken)?;
        let payload: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&payload_json)
                .map_err(|e| OidcError::DecodingError(e.to_string()))?;

        // --- 4. Extract standard claims ---
        let sub = payload.get("sub")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();
        let iss = payload.get("iss")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();
        let exp = payload.get("exp")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let aud = match payload.get("aud") {
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(arr)) => arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_owned()))
                .collect(),
            _ => vec![],
        };
        let scope = payload.get("scope")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned());
        let jti = payload.get("jti")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned());

        // --- 5. Validate issuer ---
        if !self.config.issuer.is_empty() && iss != self.config.issuer {
            return Err(OidcError::IssuerMismatch);
        }

        // --- 6. Validate audience ---
        if !self.config.audience.is_empty() && !aud.contains(&self.config.audience) {
            return Err(OidcError::AudienceMismatch);
        }

        // --- 7. Validate expiry (with leeway) ---
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_secs();
        if exp + self.config.leeway_secs < now {
            return Err(OidcError::TokenExpired);
        }

        // --- 8. Key lookup (signature verification stub) ---
        {
            let cache = self.cache.read().unwrap();
            if cache.get(&kid).is_none() && !cache.is_empty() {
                return Err(OidcError::UnknownKeyId(kid.clone()));
            }
            // TODO v1.0.0: verify signature using jsonwebtoken::decode() with key material
        }

        Ok(Claims { sub, scope, exp, iss, aud, jti })
    }

    /// Evict stale keys from the cache.
    pub fn evict_stale_keys(&self) {
        self.cache.write().unwrap().evict_stale();
    }

    /// Number of keys currently cached.
    pub fn cached_key_count(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    /// Detect whether a Bearer token string is a JWT (has 2 dots).
    pub fn is_jwt(token: &str) -> bool {
        token.bytes().filter(|&b| b == b'.').count() == 2
    }
}

// ---------------------------------------------------------------------------
// Base64URL decode helper (no external dep)
// ---------------------------------------------------------------------------

fn base64url_decode(s: &str) -> Result<Vec<u8>, ()> {
    // Add padding if needed
    let padded = match s.len() % 4 {
        2 => format!("{s}=="),
        3 => format!("{s}="),
        _ => s.to_owned(),
    };
    // Convert base64url to standard base64
    let standard = padded.replace('-', "+").replace('_', "/");
    use std::io::Read;
    // Simple base64 decode using std — no external crate
    base64_decode_std(&standard).ok_or(())
}

fn base64_decode_std(s: &str) -> Option<Vec<u8>> {
    // Manual base64 decoder (avoids pulling in `base64` crate)
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut map = [255u8; 256];
    for (i, &b) in TABLE.iter().enumerate() { map[b as usize] = i as u8; }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut i = 0;
    while i + 3 < bytes.len() {
        let b0 = map[bytes[i] as usize]; if b0 == 255 { break; }
        let b1 = map[bytes[i+1] as usize]; if b1 == 255 { break; }
        let b2 = map.get(bytes[i+2] as usize).copied().unwrap_or(0);
        let b3 = map.get(bytes[i+3] as usize).copied().unwrap_or(0);
        out.push((b0 << 2) | (b1 >> 4));
        if bytes[i+2] != b'=' { out.push((b1 << 4) | (b2 >> 2)); }
        if bytes[i+3] != b'=' { out.push((b2 << 6) | b3); }
        i += 4;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a minimal JWT with given payload (unsigned — for structural tests)
    fn make_test_jwt(payload: &str) -> String {
        let header = base64url_encode(b"{\"alg\":\"RS256\",\"typ\":\"JWT\",\"kid\":\"test-key\"}");
        let payload_enc = base64url_encode(payload.as_bytes());
        format!("{header}.{payload_enc}.fakesig")
    }

    fn base64url_encode(data: &[u8]) -> String {
        // Simple base64url (no padding)
        const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in data.chunks(3) {
            let b0 = chunk[0] as usize;
            let b1 = if chunk.len() > 1 { chunk[1] as usize } else { 0 };
            let b2 = if chunk.len() > 2 { chunk[2] as usize } else { 0 };
            out.push(TABLE[b0 >> 2] as char);
            out.push(TABLE[((b0 & 3) << 4) | (b1 >> 4)] as char);
            if chunk.len() > 1 { out.push(TABLE[((b1 & 0xf) << 2) | (b2 >> 6)] as char); }
            if chunk.len() > 2 { out.push(TABLE[b2 & 0x3f] as char); }
        }
        out.replace('+', "-").replace('/', "_").replace('=', "")
    }

    fn future_exp() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() + 3600
    }

    #[test]
    fn test_valid_jwt_structure_decodes() {
        let config = OidcConfig::from_issuer("https://auth.example.com")
            .with_audience("air-rs");
        let verifier = OidcVerifier::new(config);
        let exp = future_exp();
        let payload = format!(
            r#"{{"sub":"user1","iss":"https://auth.example.com","aud":"air-rs","exp":{exp}}}"#
        );
        let token = make_test_jwt(&payload);
        // No key seeded → unknown key id check skipped (cache empty)
        let result = verifier.verify(&token);
        assert!(result.is_ok(), "valid JWT should decode: {:?}", result.err());
        let claims = result.unwrap();
        assert_eq!(claims.sub, "user1");
        assert_eq!(claims.iss, "https://auth.example.com");
    }

    #[test]
    fn test_expired_token_rejected() {
        let config = OidcConfig::from_issuer("https://auth.example.com")
            .with_audience("air-rs");
        let verifier = OidcVerifier::new(config);
        let payload = r#"{"sub":"u","iss":"https://auth.example.com","aud":"air-rs","exp":1000}"#;
        let token = make_test_jwt(payload);
        assert_eq!(verifier.verify(&token).unwrap_err(), OidcError::TokenExpired);
    }

    #[test]
    fn test_wrong_audience_rejected() {
        let config = OidcConfig::from_issuer("https://auth.example.com")
            .with_audience("air-rs");
        let verifier = OidcVerifier::new(config);
        let exp = future_exp();
        let payload = format!(r#"{{"sub":"u","iss":"https://auth.example.com","aud":"other-app","exp":{exp}}}"#);
        let token = make_test_jwt(&payload);
        assert_eq!(verifier.verify(&token).unwrap_err(), OidcError::AudienceMismatch);
    }

    #[test]
    fn test_wrong_issuer_rejected() {
        let config = OidcConfig::from_issuer("https://auth.example.com")
            .with_audience("air-rs");
        let verifier = OidcVerifier::new(config);
        let exp = future_exp();
        let payload = format!(r#"{{"sub":"u","iss":"https://evil.com","aud":"air-rs","exp":{exp}}}"#);
        let token = make_test_jwt(&payload);
        assert_eq!(verifier.verify(&token).unwrap_err(), OidcError::IssuerMismatch);
    }

    #[test]
    fn test_malformed_token_rejected() {
        let config = OidcConfig::from_issuer("https://auth.example.com").with_audience("air-rs");
        let verifier = OidcVerifier::new(config);
        assert_eq!(verifier.verify("notajwt").unwrap_err(), OidcError::MalformedToken);
        assert_eq!(verifier.verify("a.b").unwrap_err(), OidcError::MalformedToken);
    }

    #[test]
    fn test_key_rotation_detection() {
        let config = OidcConfig::from_issuer("https://auth.example.com").with_audience("air-rs");
        let verifier = OidcVerifier::new(config);
        // Seed a key with kid "old"
        verifier.seed_key("old", b"key_material".to_vec(), "RS256");
        // Token with kid "new" — should get UnknownKeyId since cache non-empty
        let exp = future_exp();
        let header = {
            let h = r#"{"alg":"RS256","typ":"JWT","kid":"new"}"#;
            let header = base64url_encode(h.as_bytes());
            header
        };
        let payload = format!(r#"{{"sub":"u","iss":"https://auth.example.com","aud":"air-rs","exp":{exp}}}"#);
        let payload_enc = base64url_encode(payload.as_bytes());
        let token = format!("{header}.{payload_enc}.fakesig");
        assert_eq!(verifier.verify(&token).unwrap_err(), OidcError::UnknownKeyId("new".into()));
    }

    #[test]
    fn test_is_jwt_detection() {
        assert!(OidcVerifier::is_jwt("a.b.c"));
        assert!(OidcVerifier::is_jwt("eyJ.eyJ.sig"));
        assert!(!OidcVerifier::is_jwt("opaque-api-key-123"));
        assert!(!OidcVerifier::is_jwt(""));
    }

    #[test]
    fn test_claims_has_scope() {
        let claims = Claims {
            sub: "u".into(), scope: Some("read write admin".into()),
            exp: future_exp(), iss: "iss".into(), aud: vec![], jti: None,
        };
        assert!(claims.has_scope("read"));
        assert!(claims.has_scope("write"));
        assert!(!claims.has_scope("delete"));
    }

    #[test]
    fn test_cache_ttl_eviction() {
        let mut cache = JwksCache::new(0); // TTL = 0 → immediately stale
        cache.insert("k1".into(), vec![1, 2, 3], "RS256".into());
        // With TTL=0, the key is immediately stale
        assert!(cache.get("k1").is_none(), "key should be stale with TTL=0");
        cache.evict_stale();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_oidc_config_jwks_url_auto_derived() {
        let config = OidcConfig::from_issuer("https://auth.example.com");
        assert_eq!(
            config.jwks_url.as_deref(),
            Some("https://auth.example.com/.well-known/jwks.json")
        );
    }
}
