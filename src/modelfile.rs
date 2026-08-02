//! Declarative Modelfile parser for model deployment configurations.
//!
//! Supports parsing system prompts, default parameters (temperature, top_p, max_tokens),
//! stop sequences, and prompt templates.

use crate::dispatcher::GenerateConfig;
use anyhow::{Context, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Modelfile {
    pub from: Option<String>,
    pub system: Option<String>,
    pub template: Option<String>,
    pub stop: Vec<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
}

impl Default for Modelfile {
    fn default() -> Self {
        Self {
            from: None,
            system: None,
            template: None,
            stop: Vec::new(),
            temperature: None,
            top_p: None,
            max_tokens: None,
        }
    }
}

impl Modelfile {
    pub fn parse(input: &str) -> Result<Self> {
        let mut mf = Modelfile::default();

        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let mut parts = trimmed.splitn(2, |c: char| c.is_whitespace());
            let cmd = parts.next().unwrap_or("").to_uppercase();
            let rest = parts.next().unwrap_or("").trim();

            match cmd.as_str() {
                "FROM" => {
                    mf.from = Some(unquote(rest).to_string());
                }
                "SYSTEM" => {
                    mf.system = Some(unquote(rest).to_string());
                }
                "TEMPLATE" => {
                    mf.template = Some(unquote(rest).to_string());
                }
                "STOP" => {
                    let s = unquote(rest);
                    if !s.is_empty() {
                        mf.stop.push(s.to_string());
                    }
                }
                "PARAMETER" => {
                    let mut param_parts = rest.splitn(2, |c: char| c.is_whitespace());
                    let p_name = param_parts.next().unwrap_or("").to_lowercase();
                    let p_val = param_parts.next().unwrap_or("").trim();

                    match p_name.as_str() {
                        "temperature" => {
                            mf.temperature = p_val.parse().ok();
                        }
                        "top_p" => {
                            mf.top_p = p_val.parse().ok();
                        }
                        "max_tokens" | "num_predict" => {
                            mf.max_tokens = p_val.parse().ok();
                        }
                        "stop" => {
                            let s = unquote(p_val);
                            if !s.is_empty() {
                                mf.stop.push(s.to_string());
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        Ok(mf)
    }

    /// Apply Modelfile configuration defaults to a GenerateConfig request.
    pub fn apply_to(&self, config: &mut GenerateConfig) {
        if let Some(sys) = &self.system {
            if config.prompt.is_empty() {
                config.prompt = sys.clone();
            } else if !config.prompt.contains(sys) {
                config.prompt = format!("System: {}\n\nUser: {}", sys, config.prompt);
            }
        }

        if let Some(temp) = self.temperature {
            config.temperature = temp;
        }

        if let Some(top_p) = self.top_p {
            config.top_p = top_p;
        }

        if let Some(max_t) = self.max_tokens {
            config.max_tokens = max_t;
        }

        for s in &self.stop {
            if !config.stop.contains(s) {
                config.stop.push(s.clone());
            }
        }
    }
}

fn unquote(s: &str) -> &str {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        if s.len() >= 2 {
            return &s[1..s.len() - 1];
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_modelfile() {
        let content = r#"
FROM llama3
SYSTEM "You are a helpful AI coding assistant."
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER max_tokens 512
STOP "<|eot_id|>"
STOP "<|end_of_text|>"
"#;

        let mf = Modelfile::parse(content).unwrap();
        assert_eq!(mf.from, Some("llama3".to_string()));
        assert_eq!(mf.system, Some("You are a helpful AI coding assistant.".to_string()));
        assert_eq!(mf.temperature, Some(0.7));
        assert_eq!(mf.top_p, Some(0.9));
        assert_eq!(mf.max_tokens, Some(512));
        assert_eq!(mf.stop.len(), 2);
        assert_eq!(mf.stop[0], "<|eot_id|>");
    }

    #[test]
    fn test_apply_to_config() {
        let content = r#"
SYSTEM "Be precise."
PARAMETER temperature 0.2
STOP "END"
"#;
        let mf = Modelfile::parse(content).unwrap();
        let mut cfg = GenerateConfig::default();
        cfg.prompt = "Write code".to_string();

        mf.apply_to(&mut cfg);

        assert_eq!(cfg.temperature, 0.2);
        assert!(cfg.prompt.contains("Be precise."));
        assert_eq!(cfg.stop, vec!["END".to_string()]);
    }
}
