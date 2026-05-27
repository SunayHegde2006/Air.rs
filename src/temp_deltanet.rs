/// Run the Gated DeltaNet recurrent forward pass (Qwen 3.6 hybrid).
pub fn forward_deltanet(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    _weights: &QBlockWeights,
    state: &mut crate::gated_deltanet::DeltaState,
    _config: &ModelConfig,
) -> Result<Tensor> {
    let (batch, seq_len, _hidden) = q.dims3()?;
    let device = q.device();
    
    let alpha_t = 1.0f32;
    let beta_t = 0.0f32;
    
    if !matches!(device, Device::Cpu) {
        if state.tensor.is_none() {
            state.tensor = Some(Tensor::zeros((state.n_heads, state.d_v, state.d_k), candle_core::DType::F32, device)?);
        }
        let (n_heads_state, d_v_state, d_k_state) = state.tensor.as_ref().unwrap().dims3()?;
        let mut s = state.tensor.as_mut().unwrap();
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let q_t = q.i((.., t, ..))?;
            let k_t = k.i((.., t, ..))?;
            let v_t = v.i((.., t, ..))?;
            let g_t = gate.i((.., t, ..))?;

            let d = d_k_state;
            let n_q = q_t.dim(1)? / d;
            let n_kv = k_t.dim(1)? / d;
            let ratio = n_q / n_kv;
            
            let q_th = q_t.reshape((batch, n_q, d))?;
            let k_th = k_t.reshape((batch, n_kv, d))?;
            let v_th = v_t.reshape((batch, n_kv, d))?;
            let g_th = g_t.reshape((batch, n_q, d))?;

            let k_th = if ratio > 1 {
                k_th.unsqueeze(2)?.expand((batch, n_kv, ratio, d))?.reshape((batch, n_q, d))?
            } else { k_th };
            let v_th = if ratio > 1 {
                v_th.unsqueeze(2)?.expand((batch, n_kv, ratio, d))?.reshape((batch, n_q, d))?
            } else { v_th };

            let delta = crate::gated_deltanet::sigmoid_tensor(&(g_th.clone() + beta_t as f64)?)?;
            let ad = delta.affine(alpha_t as f64, 0.0)?;

            let sk = s.matmul(&k_th.i(0)?.unsqueeze(2)?)?.squeeze(2)?;
            let decay_term = sk.broadcast_mul(&ad.i(0)?)?;
            let v_minus_decay = v_th.i(0)?.sub(&decay_term)?;

            let update = v_minus_decay.unsqueeze(2)?.matmul(&k_th.i(0)?.unsqueeze(1)?)?;
            *s = s.add(&update)?;

            let o = s.matmul(&q_th.i(0)?.unsqueeze(2)?)?.squeeze(2)?;
            let gated_o = o.mul(&crate::gated_deltanet::sigmoid_tensor(&g_th.i(0)?)?)?;
            outputs.push(gated_o.unsqueeze(0)?);
        }
        
        let out = Tensor::cat(&outputs, 1)?;
        return Ok(out.reshape((batch, seq_len, ()))?);
    }

    let q_cpu = q.to_device(&Device::Cpu)?.to_dtype(candle_core::DType::F32)?;
    let k_cpu = k.to_device(&Device::Cpu)?.to_dtype(candle_core::DType::F32)?;
    let v_cpu = v.to_device(&Device::Cpu)?.to_dtype(candle_core::DType::F32)?;
    let gate_cpu = gate.to_device(&Device::Cpu)?.to_dtype(candle_core::DType::F32)?;

    let q_data = q_cpu.flatten_all()?.to_vec1::<f32>()?;
    let k_data = k_cpu.flatten_all()?.to_vec1::<f32>()?;
    let v_data = v_cpu.flatten_all()?.to_vec1::<f32>()?;
    let g_data = gate_cpu.flatten_all()?.to_vec1::<f32>()?;

    let nh = state.n_heads;
    let d = state.d_k;
    let mut out_data = vec![0.0f32; batch * seq_len * nh * d];

    for t in 0..seq_len {
        for h in 0..nh {
            let offset = (t * nh + h) * d;
            let h_q = &q_data[offset..offset + d];
            let h_k = &k_data[offset..offset + d];
            let h_v = &v_data[offset..offset + d];
            
            let h_out = crate::gated_deltanet::delta_recurrence_step_fast(
                state, h, h_q, h_k, h_v, alpha_t, beta_t
            );
            
            for (i, &val) in h_out.iter().enumerate() {
                let gate_val = g_data[offset + i];
                out_data[offset + i] = val * crate::gated_deltanet::sigmoid(gate_val);
            }
        }
    }

    Tensor::from_vec(out_data, (batch, seq_len, nh * d), &Device::Cpu)?
        .to_device(device)
}
