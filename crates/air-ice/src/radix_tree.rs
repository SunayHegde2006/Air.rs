use crate::paged_cache::PhysicalBlockId;
use std::collections::HashMap;

pub struct RadixNode {
    pub prefix_tokens: Vec<u32>,
    pub blocks: Vec<PhysicalBlockId>,
    pub children: HashMap<u32, RadixNode>,
}

impl RadixNode {
    pub fn new(prefix_tokens: Vec<u32>, blocks: Vec<PhysicalBlockId>) -> Self {
        Self {
            prefix_tokens,
            blocks,
            children: HashMap::new(),
        }
    }
}

pub struct RadixCacheTree {
    pub root: RadixNode,
}

impl RadixCacheTree {
    pub fn new() -> Self {
        Self {
            root: RadixNode::new(Vec::new(), Vec::new()),
        }
    }

    pub fn match_prefix(&self, tokens: &[u32]) -> (usize, Vec<PhysicalBlockId>) {
        let mut curr = &self.root;
        let mut matched_len = 0;
        let mut matched_blocks = Vec::new();
        let mut token_idx = 0;

        while token_idx < tokens.len() {
            let token = tokens[token_idx];
            if let Some(child) = curr.children.get(&token) {
                let common_len = child
                    .prefix_tokens
                    .iter()
                    .zip(&tokens[token_idx..])
                    .take_while(|(&a, &b)| a == b)
                    .count();

                if common_len == child.prefix_tokens.len() {
                    matched_len += common_len;
                    matched_blocks.extend_from_slice(&child.blocks);
                    token_idx += common_len;
                    curr = child;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        (matched_len, matched_blocks)
    }
}
