mod vectorstore;
use candle_core::{DType, Device, Tensor};
use candle_nn::Embedding;
use vectorstore::VectorStore;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let mut store = VectorStore::new();
    let vector = Tensor::from_vec(vec![1.0_f32; 300], &[300], &device)?;
    let query = Tensor::from_vec(vec![1.0_f32; 300], &[300], &device)?;
    store.add(vector);
    let results = store.query(&query)?;
    println!("{:?}", results);
    Ok(())
}
