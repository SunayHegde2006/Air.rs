pub struct AttentionSinkManager {
    num_sink_tokens: usize,
    sliding_window_size: usize,
}

impl AttentionSinkManager {
    pub fn new(num_sink_tokens: usize, sliding_window_size: usize) -> Self {
        Self {
            num_sink_tokens,
            sliding_window_size,
        }
    }

    pub fn should_evict(&self, seq_len: usize) -> bool {
        seq_len > self.num_sink_tokens + self.sliding_window_size
    }

    pub fn select_active_token_indices(&self, total_seq_len: usize) -> Vec<usize> {
        if total_seq_len <= self.num_sink_tokens + self.sliding_window_size {
            return (0..total_seq_len).collect();
        }

        let window_start = total_seq_len - self.sliding_window_size;
        (0..self.num_sink_tokens)
            .chain(window_start..total_seq_len)
            .collect()
    }
}
