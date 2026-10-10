use crate::error::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct ChatTemplateFormatter;

impl ChatTemplateFormatter {
    pub fn format(messages: &[ChatMessage], system_override: Option<&str>) -> Result<String> {
        let mut out = String::new();

        if let Some(sys) = system_override {
            out.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
        }

        for msg in messages {
            out.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }

        out.push_str("<|im_start|>assistant\n");
        Ok(out)
    }
}
