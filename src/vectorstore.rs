//candle embedding
use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module};
use std::f32;

pub struct VectorStore {
    embeddings: Vec<Tensor>,
}

impl VectorStore {
    pub fn new() -> Self {
        let embeddings = Vec::new();
        VectorStore { embeddings }
    }

    pub fn add(&mut self, embedding: Tensor) {
        self.embeddings.push(embedding);
    }

    fn cosine_similarity(&self, embedding: &Tensor, query: &Tensor) -> Result<f32> {
        // Reshape vectors to 2D for matrix multiplication
        let embedding = embedding.unsqueeze(0)?;
        let query = query.unsqueeze(1)?;

        let dot_product = (&embedding).matmul(&query)?;
        let norm_a = embedding.sqr()?.sum_all()?.sqrt()?.reshape(&[1, 1])?;
        let norm_b = query.sqr()?.sum_all()?.sqrt()?.reshape(&[1, 1])?;
        let denominator = (&norm_a * &norm_b)?;

        dot_product
            .div(&denominator)?
            .squeeze(0)?
            .squeeze(0)?
            .to_scalar::<f32>()
    }

    pub fn query(&self, query: &Tensor) -> Result<Vec<Tensor>> {
        let mut results = Vec::new();
        for embedding in &self.embeddings {
            let similarity = self.cosine_similarity(embedding, query)?;
            if similarity > 0.8 {
                // Threshold can be adjusted
                results.push(embedding.clone());
            }
        }
        Ok(results)
    }
}
