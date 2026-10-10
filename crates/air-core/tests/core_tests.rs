use air_core::types::Shape;
use air_core::ops::math::{rms_norm, softmax};

#[test]
fn test_shape() {
    let shape = Shape::new(vec![1, 32, 128]);
    assert_eq!(shape.rank(), 3);
    assert_eq!(shape.num_elements(), 4096);
}

#[test]
fn test_rms_norm() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let mut out = vec![0.0; 4];
    rms_norm(&input, &weight, &mut out, 1e-5);
    assert!((out[0] - 0.365148).abs() < 1e-3);
}

#[test]
fn test_softmax() {
    let mut x = vec![1.0, 2.0, 3.0];
    softmax(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_dot_product() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let res: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    assert_eq!(res, 32.0);
}
