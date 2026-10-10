#[derive(Debug, Clone)]
pub struct StopSequenceMatcher {
    sequences: Vec<Vec<u32>>,
    buffer: Vec<u32>,
}

impl StopSequenceMatcher {
    pub fn new(sequences: Vec<Vec<u32>>) -> Self {
        Self {
            sequences,
            buffer: Vec::new(),
        }
    }

    pub fn push(&mut self, token_id: u32) -> bool {
        self.buffer.push(token_id);

        for seq in &self.sequences {
            if seq.is_empty() {
                continue;
            }
            if self.buffer.ends_with(seq) {
                return true;
            }
        }

        false
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}
