//candle embedding
use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module};
use std::f32;

pub struct VectorStore {
    embeddings: Vec<Tensor>,
    texts: Vec<String>, // Store original texts
}

impl VectorStore {
    pub fn new() -> Self {
        VectorStore {
            embeddings: Vec::new(),
            texts: Vec::new(),
        }
    }

    pub fn add(&mut self, embedding: Tensor, text: String) {
        self.embeddings.push(embedding);
        self.texts.push(text);
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

    pub fn query(&self, query_embedding: &Tensor) -> Result<Vec<(String, f32)>> {
        let mut results = Vec::new();
        for (i, embedding) in self.embeddings.iter().enumerate() {
            let similarity = self.cosine_similarity(embedding, query_embedding)?;
            results.push((self.texts[i].clone(), similarity));
        }
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(results)
    }
}
