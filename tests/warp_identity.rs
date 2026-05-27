#[cfg(test)]
mod verification {
    use air_rs::gated_deltanet::*;
    
    #[test]
    fn test_warp_numerical_identity_handoff() {
        let d = 16;
        let nh = 4;
        let seq1_len = 32;
        let seq2_len = 32;
        let cfg = DeltaNetConfig::new(d, nh);
        
        // Node A (Prefill)
        let mut node_a = GatedDeltaNetLayer::new(cfg.clone());
        let qkvab1 = vec![0.1f32; seq1_len * nh * (3*d + 2)];
        let _out_a = node_a.forward_chunk(&qkvab1, seq1_len);
        
        // Export state from Node A (as if sending over W.A.R.P.)
        let state_a = node_a.states[0].clone();
        let norm_a = state_a.frob_norm();
        
        // Node B (Decode)
        let mut node_b = GatedDeltaNetLayer::new(cfg.clone());
        // Import state into Node B
        node_b.states[0] = state_a;
        
        // Verify identity before continuing
        assert!((node_b.states[0].frob_norm() - norm_a).abs() < 1e-6);
        
        // Continue generation on Node B
        let qkvab2 = vec![0.2f32; seq2_len * nh * (3*d + 2)];
        let _out_b = node_b.forward_chunk(&qkvab2, seq2_len);
        
        // Verify final norm on Node B matches what Node A *would* have produced
        let mut node_a_ext = node_a;
        let _out_a_ext = node_a_ext.forward_chunk(&qkvab2, seq2_len);
        
        let norm_final_a = node_a_ext.states[0].frob_norm();
        let norm_final_b = node_b.states[0].frob_norm();
        
        println!("Final Norm Node A: {:.10}", norm_final_a);
        println!("Final Norm Node B: {:.10}", norm_final_b);
        
        assert!((norm_final_a - norm_final_b).abs() < 1e-7, "Numerical drift detected during W.A.R.P. handoff!");
    }
}
