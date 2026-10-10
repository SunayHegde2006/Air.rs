use air_engine::pipeline::ExecutionEngine;
use air_engine::sampler::SamplingParams;

#[test]
fn test_execution_engine_step() {
    let mut engine = ExecutionEngine::new(100, 16);
    let mut history = vec![1, 2, 3];
    let params = SamplingParams::default();
    let token = engine.step(&mut history, &params).unwrap();
    assert_eq!(history.len(), 4);
    assert_eq!(history.last(), Some(&token));
}
