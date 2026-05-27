//! Test for GBNF Tree Attention Mask integration.

use air_rs::gbnf::GbnfConstraint;
use air_rs::eagle2::{DraftTree, DraftNode};
use air_rs::tokenizer::Tokenizer;
use candle_core::Device;

#[test]
fn test_gbnf_tree_mask_pruning() {
    let dev = Device::Cpu;
    
    // 1. Setup simple vocabulary and tokenizer
    let vocab = vec!["root".to_string(), "yes".to_string(), "no".to_string(), "!".to_string()];
    let tokenizer = Tokenizer::new(vocab, vec![], 0, 0);
    
    // 2. Setup GBNF constraint: root ::= "yes" "!"
    let gbnf = GbnfConstraint::from_str("root ::= \"yes\" \"!\"", vec!["root".into(), "yes".into(), "no".into(), "!".into()]).unwrap();
    
    // 3. Build a draft tree:
    // root (ID 0)
    //   ├── yes (ID 1) [VALID]
    //   │     └── ! (ID 3) [VALID]
    //   └── no (ID 2) [INVALID]
    
    let mut tree = DraftTree::new(0);
    // Node 0: yes
    tree.nodes.push(DraftNode { token_id: 1, draft_prob: 0.9, depth: 1, parent_idx: None });
    // Node 1: no (INVALID according to grammar)
    tree.nodes.push(DraftNode { token_id: 2, draft_prob: 0.5, depth: 1, parent_idx: None });
    // Node 2: ! (child of yes) [VALID]
    tree.nodes.push(DraftNode { token_id: 3, draft_prob: 0.9, depth: 2, parent_idx: Some(0) });
    
    // 4. Generate mask
    let mask = tree.to_gbnf_mask(&dev, Some(&gbnf), Some(&tokenizer)).unwrap();
    let mask_data = mask.flatten_all().unwrap().to_vec1::<u8>().unwrap();
    let n = 3; // tree size
    let m = |i: usize, j: usize| mask_data[i * n + j];
    
    // Node 0 (yes) should be valid
    assert_eq!(m(0, 0), 1, "yes should see itself");
    
    // Node 1 (no) should be MASKED (all 0s)
    assert_eq!(m(1, 0), 0, "no row should be 0");
    assert_eq!(m(1, 1), 0, "no row should be 0");
    assert_eq!(m(1, 2), 0, "no row should be 0");
    
    // Node 2 (!) should be valid and see parent 0 (yes)
    assert_eq!(m(2, 2), 1, "! should see itself");
    assert_eq!(m(2, 0), 1, "! should see its parent (yes)");
    
    println!("✅ GBNF Tree Attention Mask verification passed.");
}
