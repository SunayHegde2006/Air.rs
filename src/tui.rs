//! Terminal User Interface for Air.rs interactive chat.
//!
//! Provides a rich terminal experience with:
//! - Streaming token display with word-wrap
//! - GPU utilisation and throughput stats in a status bar
//! - Coloured roles (system=yellow, user=green, assistant=cyan)
//! - Input line-editing with history recall
//! - Markdown-lite rendering (bold, code blocks)

use std::io::{self, Write};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// ANSI Colour Codes
// ---------------------------------------------------------------------------

/// ANSI escape sequences for terminal colouring.
pub mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const ITALIC: &str = "\x1b[3m";
    pub const UNDERLINE: &str = "\x1b[4m";

    // Foreground colours
    pub const RED: &str = "\x1b[31m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const CYAN: &str = "\x1b[36m";
    pub const WHITE: &str = "\x1b[37m";
    pub const GRAY: &str = "\x1b[90m";

    // Background colours
    pub const BG_BLACK: &str = "\x1b[40m";
    pub const BG_BLUE: &str = "\x1b[44m";
    pub const BG_GRAY: &str = "\x1b[100m";

    /// Coloured string helper.
    pub fn colour(text: &str, colour_code: &str) -> String {
        format!("{}{}{}", colour_code, text, RESET)
    }
}

// ---------------------------------------------------------------------------
// Status Bar
// ---------------------------------------------------------------------------

/// Live status bar showing inference metrics.
pub struct StatusBar {
    model_name: String,
    context_used: usize,
    context_max: usize,
    tokens_per_sec: f64,
    gpu_usage_pct: f64,
    gpu_memory_used_mb: u64,
    gpu_memory_total_mb: u64,
}

impl StatusBar {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            context_used: 0,
            context_max: 0,
            tokens_per_sec: 0.0,
            gpu_usage_pct: 0.0,
            gpu_memory_used_mb: 0,
            gpu_memory_total_mb: 0,
        }
    }

    /// Update context window fill.
    pub fn set_context(&mut self, used: usize, max: usize) {
        self.context_used = used;
        self.context_max = max;
    }

    /// Update throughput reading.
    pub fn set_throughput(&mut self, tok_per_sec: f64) {
        self.tokens_per_sec = tok_per_sec;
    }

    /// Update GPU usage.
    pub fn set_gpu(&mut self, usage_pct: f64, used_mb: u64, total_mb: u64) {
        self.gpu_usage_pct = usage_pct;
        self.gpu_memory_used_mb = used_mb;
        self.gpu_memory_total_mb = total_mb;
    }

    /// Render the status bar to a single line.
    pub fn render(&self) -> String {
        let ctx_pct = if self.context_max > 0 {
            (self.context_used as f64 / self.context_max as f64 * 100.0) as u32
        } else {
            0
        };

        let gpu_mem = if self.gpu_memory_total_mb > 0 {
            format!(
                "VRAM {}/{}MB",
                self.gpu_memory_used_mb, self.gpu_memory_total_mb
            )
        } else {
            "VRAM N/A".to_string()
        };

        format!(
            "{} {} {} | ctx {}/{} ({}%) | {:.1} t/s | GPU {:.0}% | {} {}",
            ansi::BG_GRAY,
            ansi::BOLD,
            self.model_name,
            self.context_used,
            self.context_max,
            ctx_pct,
            self.tokens_per_sec,
            self.gpu_usage_pct,
            gpu_mem,
            ansi::RESET,
        )
    }

    /// Print the status bar, overwriting the current line.
    pub fn print(&self) {
        print!("\r{}", self.render());
        io::stdout().flush().ok();
    }
}

// ---------------------------------------------------------------------------
// Streaming Token Display
// ---------------------------------------------------------------------------

/// Manages streaming token output with live display.
pub struct StreamDisplay {
    /// Terminal width for word wrapping.
    pub term_width: usize,
    /// Current column position (for word wrap).
    col: usize,
    /// Whether we're inside a code block.
    in_code_block: bool,
    /// Token counter.
    token_count: usize,
    /// Timer for throughput calculation.
    start: Instant,
}

impl StreamDisplay {
    pub fn new() -> Self {
        let width = terminal_width().unwrap_or(80);
        Self {
            term_width: width,
            col: 0,
            in_code_block: false,
            token_count: 0,
            start: Instant::now(),
        }
    }

    /// Print the beginning of an assistant response.
    pub fn start_response(&mut self) {
        println!();
        print!(
            "{}{}Assistant:{} ",
            ansi::BOLD, ansi::CYAN, ansi::RESET
        );
        io::stdout().flush().ok();
        self.col = 11; // "Assistant: " length
        self.token_count = 0;
        self.start = Instant::now();
    }

    /// Stream a single token to the display.
    pub fn push_token(&mut self, token: &str) {
        self.token_count += 1;

        // Check for code block markers
        if token.contains("```") {
            self.in_code_block = !self.in_code_block;
            if self.in_code_block {
                print!("\n{}{}", ansi::BG_GRAY, ansi::WHITE);
            } else {
                print!("{}\n", ansi::RESET);
                self.col = 0;
            }
            io::stdout().flush().ok();
            return;
        }

        // Simple word-wrap
        let new_col = self.col + token.len();
        if new_col > self.term_width && !self.in_code_block && self.col > 0 {
            println!();
            self.col = 0;
        }

        print!("{}", token);
        io::stdout().flush().ok();

        // Track column for newlines in the token
        if let Some(last_newline) = token.rfind('\n') {
            self.col = token.len() - last_newline - 1;
        } else {
            self.col += token.len();
        }
    }

    /// End the response and print stats.
    pub fn end_response(&mut self) -> GenerationStats {
        if self.in_code_block {
            print!("{}", ansi::RESET);
            self.in_code_block = false;
        }

        let elapsed = self.start.elapsed();
        let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
            self.token_count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        println!();
        println!(
            "{}[{} tokens in {:.1}s = {:.1} t/s]{}",
            ansi::DIM,
            self.token_count,
            elapsed.as_secs_f64(),
            tok_per_sec,
            ansi::RESET,
        );

        self.col = 0;

        GenerationStats {
            token_count: self.token_count,
            elapsed,
            tokens_per_sec: tok_per_sec,
        }
    }

    /// Current generation throughput.
    pub fn current_throughput(&self) -> f64 {
        let elapsed = self.start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.token_count as f64 / elapsed
        } else {
            0.0
        }
    }
}

impl Default for StreamDisplay {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from a generation run.
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub token_count: usize,
    pub elapsed: Duration,
    pub tokens_per_sec: f64,
}

// ---------------------------------------------------------------------------
// Input Prompt
// ---------------------------------------------------------------------------

/// Read a line of user input with a coloured prompt.
pub fn read_user_input() -> Option<String> {
    print!(
        "\n{}{}You:{} ",
        ansi::BOLD, ansi::GREEN, ansi::RESET
    );
    io::stdout().flush().ok();

    let mut input = String::new();
    match io::stdin().read_line(&mut input) {
        Ok(0) => None, // EOF
        Ok(_) => {
            let trimmed = input.trim().to_string();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        }
        Err(_) => None,
    }
}

/// Print a system message.
pub fn print_system(msg: &str) {
    println!(
        "{}{}System:{} {}",
        ansi::BOLD, ansi::YELLOW, ansi::RESET, msg
    );
}

/// Print an error message.
pub fn print_error(msg: &str) {
    eprintln!(
        "{}{}Error:{} {}",
        ansi::BOLD, ansi::RED, ansi::RESET, msg
    );
}

/// Print an info message.
pub fn print_info(msg: &str) {
    println!(
        "{}{}Info:{} {}",
        ansi::BOLD, ansi::BLUE, ansi::RESET, msg
    );
}

// ---------------------------------------------------------------------------
// Welcome Banner
// ---------------------------------------------------------------------------

/// Print the Air.rs welcome banner.
pub fn print_banner(model_name: &str, params: &str, quant: &str) {
    println!();
    println!(
        "{}{}  Air.rs — Local LLM Inference Engine  {}",
        ansi::BG_BLUE, ansi::BOLD, ansi::RESET
    );
    println!();
    println!(
        "  Model:  {}{}{}{}",
        ansi::BOLD, ansi::WHITE, model_name, ansi::RESET
    );
    println!(
        "  Params: {}{}{}{}  Quant: {}{}{}{}",
        ansi::BOLD, ansi::CYAN, params, ansi::RESET,
        ansi::BOLD, ansi::CYAN, quant, ansi::RESET,
    );
    println!();
    println!(
        "  {}Commands: /help, /clear, /stats, /exit{}",
        ansi::DIM, ansi::RESET
    );
    println!(
        "  {}─────────────────────────────────────────{}",
        ansi::DIM, ansi::RESET
    );
    println!();
}

// ---------------------------------------------------------------------------
// Slash Commands
// ---------------------------------------------------------------------------

/// A slash command parsed from user input.
#[derive(Debug, PartialEq)]
pub enum SlashCommand {
    Help,
    Clear,
    Stats,
    Exit,
    Context,
    Model(String),
    Temperature(f32),
    Unknown(String),
}

/// Parse a slash command from input text.
pub fn parse_command(input: &str) -> Option<SlashCommand> {
    if !input.starts_with('/') {
        return None;
    }

    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0].to_lowercase();
    let arg = parts.get(1).map(|s| s.trim().to_string());

    Some(match cmd.as_str() {
        "/help" | "/h" | "/?" => SlashCommand::Help,
        "/clear" | "/c" => SlashCommand::Clear,
        "/stats" | "/s" => SlashCommand::Stats,
        "/exit" | "/quit" | "/q" => SlashCommand::Exit,
        "/context" | "/ctx" => SlashCommand::Context,
        "/model" => {
            if let Some(name) = arg {
                SlashCommand::Model(name)
            } else {
                SlashCommand::Help
            }
        }
        "/temp" | "/temperature" => {
            if let Some(val) = arg.and_then(|s| s.parse::<f32>().ok()) {
                SlashCommand::Temperature(val)
            } else {
                SlashCommand::Help
            }
        }
        other => SlashCommand::Unknown(other.to_string()),
    })
}

/// Print help text for slash commands.
pub fn print_help() {
    println!();
    println!("{}{}Available Commands:{}", ansi::BOLD, ansi::CYAN, ansi::RESET);
    println!("  /help, /h       Show this help message");
    println!("  /clear, /c      Clear conversation history");
    println!("  /stats, /s      Show generation statistics");
    println!("  /context, /ctx  Show context window usage");
    println!("  /model <name>   Switch model");
    println!("  /temp <value>   Set temperature (0.0-2.0)");
    println!("  /exit, /quit    Exit Air.rs");
    println!();
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Get the terminal width, or None if it can't be determined.
fn terminal_width() -> Option<usize> {
    // Try environment variable first
    if let Ok(cols) = std::env::var("COLUMNS") {
        if let Ok(w) = cols.parse::<usize>() {
            return Some(w);
        }
    }
    // Default fallback
    Some(80)
}

/// Clear the terminal screen.
pub fn clear_screen() {
    print!("\x1b[2J\x1b[H");
    io::stdout().flush().ok();
}

/// Move cursor to the beginning of the current line.
pub fn carriage_return() {
    print!("\r");
    io::stdout().flush().ok();
}

/// Erase the current line.
pub fn erase_line() {
    print!("\x1b[2K\r");
    io::stdout().flush().ok();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ansi_colour() {
        let coloured = ansi::colour("hello", ansi::RED);
        assert!(coloured.contains("\x1b[31m"));
        assert!(coloured.contains("\x1b[0m"));
        assert!(coloured.contains("hello"));
    }

    #[test]
    fn test_parse_command_help() {
        assert_eq!(parse_command("/help"), Some(SlashCommand::Help));
        assert_eq!(parse_command("/h"), Some(SlashCommand::Help));
        assert_eq!(parse_command("/?"), Some(SlashCommand::Help));
    }

    #[test]
    fn test_parse_command_exit() {
        assert_eq!(parse_command("/exit"), Some(SlashCommand::Exit));
        assert_eq!(parse_command("/quit"), Some(SlashCommand::Exit));
        assert_eq!(parse_command("/q"), Some(SlashCommand::Exit));
    }

    #[test]
    fn test_parse_command_temperature() {
        assert_eq!(
            parse_command("/temp 0.7"),
            Some(SlashCommand::Temperature(0.7))
        );
    }

    #[test]
    fn test_parse_command_model() {
        assert_eq!(
            parse_command("/model llama3"),
            Some(SlashCommand::Model("llama3".to_string()))
        );
    }

    #[test]
    fn test_parse_command_unknown() {
        assert_eq!(
            parse_command("/foo"),
            Some(SlashCommand::Unknown("/foo".to_string()))
        );
    }

    #[test]
    fn test_parse_command_not_command() {
        assert_eq!(parse_command("hello"), None);
        assert_eq!(parse_command(""), None);
    }

    #[test]
    fn test_status_bar_render() {
        let mut bar = StatusBar::new("Llama-3-8B-Q4");
        bar.set_context(512, 8192);
        bar.set_throughput(42.5);
        bar.set_gpu(75.0, 4096, 8192);
        let rendered = bar.render();
        assert!(rendered.contains("Llama-3-8B-Q4"));
        assert!(rendered.contains("512/8192"));
        assert!(rendered.contains("42.5 t/s"));
    }

    #[test]
    fn test_stream_display_throughput() {
        let display = StreamDisplay::new();
        let tp = display.current_throughput();
        // No tokens pushed, should be 0
        assert_eq!(tp, 0.0);
    }

    #[test]
    fn test_terminal_width_default() {
        let width = terminal_width();
        assert!(width.is_some());
        assert!(width.unwrap() >= 40);
    }
}
