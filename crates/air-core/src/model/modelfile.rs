use crate::error::{AirError, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct Modelfile {
    pub from_model: String,
    pub system_prompt: Option<String>,
    pub template: Option<String>,
    pub parameters: HashMap<String, String>,
}

impl Modelfile {
    pub fn parse_str(content: &str) -> Result<Self> {
        let mut modelfile = Modelfile::default();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let mut parts = trimmed.splitn(2, ' ');
            let command = parts.next().unwrap_or("").to_uppercase();
            let value = parts.next().unwrap_or("").trim();

            match command.as_str() {
                "FROM" => modelfile.from_model = value.trim_matches('"').to_string(),
                "SYSTEM" => modelfile.system_prompt = Some(value.trim_matches('"').to_string()),
                "TEMPLATE" => modelfile.template = Some(value.trim_matches('"').to_string()),
                "PARAMETER" => {
                    let mut param_parts = value.splitn(2, ' ');
                    let key = param_parts.next().unwrap_or("").to_string();
                    let val = param_parts.next().unwrap_or("").to_string();
                    modelfile.parameters.insert(key, val);
                }
                _ => {}
            }
        }

        if modelfile.from_model.is_empty() {
            return Err(AirError::Other("Modelfile missing FROM directive".to_string()));
        }

        Ok(modelfile)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::parse_str(&content)
    }
}
