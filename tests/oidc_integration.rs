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
                "n": "qflgXqk2lp3dEFFcPb3GZf3YDhy0jYVQX8LC2Fs8ku5uk2GrFRqxqcm9VVLEsLiATQ5jDBSI5c3LgQHMNjFDoiqb358nvlB9zpMhRiC-HH_2GGTTNNSxqzg_sl3a2cvCa0oy8AJdvdFQB4U_r44kNCTNLsNs67slcTLt4XUB4cr-2C3JKVlp4TdvcpI5JWaaWoVpw60KhIvgnbG6Z_XLWhGVnrlTNyh2_m1Dq3Y9fWg_jCuwI9OKR3RysZ3OxeZf7_LOFQ4eo34QHrDyugeh1cXe6mnCEp44479N2ciDT3E6WDWXcnzsoRiAjCf6qFjbnl3eB1g5sO49FNUT0JhYqw",
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
