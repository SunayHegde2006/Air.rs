use air_rs::oidc::{OidcConfig, OidcVerifier};
use httpmock::prelude::*;
use serde_json::json;

#[test]
fn test_oidc_fetch_jwks_from_mock_server() {
    // 1. Start mock server
    let server = MockServer::start();

    // 2. Define a mock JWKS response
    let jwks_response = json!({
        "keys": [
            {
                "kty": "RSA",
                "kid": "test-key-1",
                "use": "sig",
                "alg": "RS256",
                "n": "v7_Vp-K...placeholder",
                "e": "AQAB"
            }
        ]
    });

    let jwks_mock = server.mock(|when, then| {
        when.method(GET)
            .path("/.well-known/jwks.json");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(jwks_response);
    });

    // 3. Configure OidcVerifier to point to mock server
    let config = OidcConfig {
        issuer: server.base_url(),
        audience: "air-rs".into(),
        jwks_url: Some(server.url("/.well-known/jwks.json")),
        cache_ttl_secs: 300,
        leeway_secs: 30,
    };
    let verifier = OidcVerifier::new(config);

    // 4. Trigger fetch
    let result = verifier.fetch_jwks();

    // 5. Assertions
    assert!(result.is_ok(), "fetch_jwks should succeed with mock server");
    jwks_mock.assert();
    assert_eq!(verifier.cached_key_count(), 1, "Should have 1 key cached after fetch");
}
