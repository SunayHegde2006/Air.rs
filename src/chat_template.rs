//! Chat Template Support for LLM Inference.
//!
//! Different models expect different prompt formats. This module provides
//! template formatting for the most common chat formats, plus auto-detection
//! from GGUF metadata.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Message Types
// ---------------------------------------------------------------------------

/// Role in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
        }
    }
}

impl Role {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            _ => None,
        }
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

// ---------------------------------------------------------------------------
// Chat Formats
// ---------------------------------------------------------------------------

/// Supported chat template formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatFormat {
    ChatML,
    Llama3,
    Mistral,
    Phi3,
    Gemma,
    Raw,
}

impl std::fmt::Display for ChatFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatFormat::ChatML => write!(f, "chatml"),
            ChatFormat::Llama3 => write!(f, "llama3"),
            ChatFormat::Mistral => write!(f, "mistral"),
            ChatFormat::Phi3 => write!(f, "phi3"),
            ChatFormat::Gemma => write!(f, "gemma"),
            ChatFormat::Raw => write!(f, "raw"),
        }
    }
}

/// Apply a chat template to a list of messages.
pub struct ChatTemplate {
    pub format: ChatFormat,
    pub bos_token: String,
    pub eos_token: String,
    pub add_generation_prompt: bool,
}

impl ChatTemplate {
    /// Create a template for the given format.
    pub fn new(format: ChatFormat) -> Self {
        let (bos, eos) = match &format {
            ChatFormat::ChatML => ("<|im_start|>".into(), "<|im_end|>".into()),
            ChatFormat::Llama3 => ("<|begin_of_text|>".into(), "<|eot_id|>".into()),
            ChatFormat::Mistral => ("<s>".into(), "".into()),
            ChatFormat::Phi3 => ("".into(), "<|end|>".into()),
            ChatFormat::Gemma => ("<bos>".into(), "<eos>".into()),
            ChatFormat::Raw => ("".into(), "".into()),
        };

        Self {
            format,
            bos_token: bos,
            eos_token: eos,
            add_generation_prompt: true,
        }
    }

    /// Apply the template to a list of messages, producing a formatted prompt.
    pub fn apply(&self, messages: &[ChatMessage]) -> String {
        match self.format {
            ChatFormat::ChatML => self.apply_chatml(messages),
            ChatFormat::Llama3 => self.apply_llama3(messages),
            ChatFormat::Mistral => self.apply_mistral(messages),
            ChatFormat::Phi3 => self.apply_phi3(messages),
            ChatFormat::Gemma => self.apply_gemma(messages),
            ChatFormat::Raw => self.apply_raw(messages),
        }
    }

    fn apply_chatml(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            out.push_str("<|im_start|>");
            out.push_str(&format!("{}\n{}\n", msg.role, msg.content));
            out.push_str("<|im_end|>\n");
        }
        if self.add_generation_prompt {
            out.push_str("<|im_start|>assistant\n");
        }
        out
    }

    fn apply_llama3(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::from("<|begin_of_text|>");
        for msg in messages {
            out.push_str("<|start_header_id|>");
            out.push_str(&msg.role.to_string());
            out.push_str("<|end_header_id|>\n\n");
            out.push_str(&msg.content);
            out.push_str("<|eot_id|>");
        }
        if self.add_generation_prompt {
            out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        out
    }

    fn apply_mistral(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::from("<s>");
        let mut pending_user: Option<&str> = None;

        for msg in messages {
            match msg.role {
                Role::System | Role::User => {
                    if let Some(prev) = pending_user {
                        out.push_str(&format!("[INST] {} [/INST]", prev));
                    }
                    pending_user = Some(&msg.content);
                }
                Role::Assistant => {
                    if let Some(user_msg) = pending_user.take() {
                        out.push_str(&format!("[INST] {} [/INST] {} ", user_msg, msg.content));
                    }
                }
            }
        }
        if let Some(user_msg) = pending_user {
            out.push_str(&format!("[INST] {} [/INST]", user_msg));
        }
        out
    }

    fn apply_phi3(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            let tag = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            out.push_str(&format!("<|{}|>\n{}<|end|>\n", tag, msg.content));
        }
        if self.add_generation_prompt {
            out.push_str("<|assistant|>\n");
        }
        out
    }

    fn apply_gemma(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            let tag = match msg.role {
                Role::System | Role::User => "user",
                Role::Assistant => "model",
            };
            out.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", tag, msg.content));
        }
        if self.add_generation_prompt {
            out.push_str("<start_of_turn>model\n");
        }
        out
    }

    fn apply_raw(&self, messages: &[ChatMessage]) -> String {
        messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n")
    }
}

/// Attempt to detect the chat format from GGUF metadata or model name.
pub fn detect_format(model_name: &str, metadata: &HashMap<String, String>) -> ChatFormat {
    // Check GGUF metadata first
    if let Some(template) = metadata.get("tokenizer.chat_template") {
        let t = template.to_lowercase();
        if t.contains("im_start") { return ChatFormat::ChatML; }
        if t.contains("start_header_id") { return ChatFormat::Llama3; }
        if t.contains("[inst]") { return ChatFormat::Mistral; }
        if t.contains("start_of_turn") { return ChatFormat::Gemma; }
    }

    // Fall back to model name heuristics
    let name = model_name.to_lowercase();
    if name.contains("llama-3") || name.contains("llama3") { return ChatFormat::Llama3; }
    if name.contains("mistral") || name.contains("mixtral") { return ChatFormat::Mistral; }
    if name.contains("phi-3") || name.contains("phi3") { return ChatFormat::Phi3; }
    if name.contains("gemma") { return ChatFormat::Gemma; }
    if name.contains("qwen") || name.contains("yi-") { return ChatFormat::ChatML; }

    ChatFormat::ChatML // Safe default
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_format() {
        let tpl = ChatTemplate::new(ChatFormat::ChatML);
        let msgs = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello!"),
        ];
        let result = tpl.apply(&msgs);
        assert!(result.contains("<|im_start|>"));
        assert!(result.contains("You are helpful."));
        assert!(result.contains("assistant"));
    }

    #[test]
    fn test_llama3_format() {
        let tpl = ChatTemplate::new(ChatFormat::Llama3);
        let msgs = vec![ChatMessage::user("Hi there")];
        let result = tpl.apply(&msgs);
        assert!(result.contains("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>"));
        assert!(result.contains("Hi there"));
    }

    #[test]
    fn test_mistral_format() {
        let tpl = ChatTemplate::new(ChatFormat::Mistral);
        let msgs = vec![
            ChatMessage::user("What is 2+2?"),
            ChatMessage::assistant("4"),
            ChatMessage::user("And 3+3?"),
        ];
        let result = tpl.apply(&msgs);
        assert!(result.starts_with("<s>"));
        assert!(result.contains("[INST]"));
    }

    #[test]
    fn test_detect_format_metadata() {
        let mut meta = HashMap::new();
        meta.insert("tokenizer.chat_template".into(), "im_start template".into());
        assert_eq!(detect_format("unknown", &meta), ChatFormat::ChatML);
    }

    #[test]
    fn test_detect_format_name() {
        let meta = HashMap::new();
        assert_eq!(detect_format("Meta-Llama-3-8B", &meta), ChatFormat::Llama3);
        assert_eq!(detect_format("Mistral-7B-v0.3", &meta), ChatFormat::Mistral);
        assert_eq!(detect_format("Phi-3-mini", &meta), ChatFormat::Phi3);
    }

    #[test]
    fn test_raw_format() {
        let tpl = ChatTemplate::new(ChatFormat::Raw);
        let msgs = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("World"),
        ];
        let result = tpl.apply(&msgs);
        assert!(result.contains("Hello"));
        assert!(result.contains("World"));
    }

    #[test]
    fn test_role_display() {
        assert_eq!(format!("{}", Role::System), "system");
        assert_eq!(format!("{}", Role::User), "user");
        assert_eq!(format!("{}", Role::Assistant), "assistant");
    }

    #[test]
    fn test_role_parse() {
        assert_eq!(Role::parse("System"), Some(Role::System));
        assert_eq!(Role::parse("USER"), Some(Role::User));
        assert_eq!(Role::parse("unknown"), None);
    }
}
