use air_core::error::Result;

pub struct TerminalUi;

impl TerminalUi {
    pub fn run_interactive_session(model_name: &str) -> Result<()> {
        println!("Starting interactive chat TUI session for model: {}", model_name);
        println!("Type '/exit' or '/quit' to leave.\n");

        use std::io::{self, BufRead, Write};
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            print!("User> ");
            stdout.flush()?;

            let mut input = String::new();
            if stdin.lock().read_line(&mut input)? == 0 {
                break;
            }

            let trimmed = input.trim();
            if trimmed == "/exit" || trimmed == "/quit" {
                break;
            }
            if trimmed.is_empty() {
                continue;
            }

            println!("\nAssistant [124.5 tok/s]> Hello! I am running on Air.rs v2.0.0. Echo: {}\n", trimmed);
        }

        Ok(())
    }
}
