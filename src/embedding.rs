use anyhow::Result;
use candle_core::{Device, Tensor};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};

pub struct Embeddor {
    model: SentenceEmbeddingsModel,
    device: Device,
}

impl Embeddor {
    pub fn new(model_type: SentenceEmbeddingsModelType) -> Result<Self> {
        let model: SentenceEmbeddingsModel =
            SentenceEmbeddingsBuilder::remote(model_type).create_model()?;
        let device = Device::Cpu;
        Ok(Self { model, device })
    }

    pub fn chunk(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut current_chunk = String::new();
        for word in words {
            if current_chunk.is_empty() {
                current_chunk.push_str(word);
            } else if current_chunk.len() + word.len() + 1 <= 512 {
                current_chunk.push(' ');
                current_chunk.push_str(word);
            } else {
                chunks.push(current_chunk);
                current_chunk = String::new();
                current_chunk.push_str(word);
            }
        }
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }
        chunks
    }

    pub fn embed(&self, text: &str) -> Result<Vec<Tensor>> {
        let chunks = self.chunk(text);
        let embeddings = self.model.encode(&chunks)?;

        let tensors = embeddings
            .iter()
            .map(|embedding| {
                Tensor::from_vec(embedding.clone(), &[embedding.len()], &self.device)
                    .map_err(anyhow::Error::from)
            })
            .collect::<Result<Vec<Tensor>>>()?;

        Ok(tensors)
    }

    pub fn decode(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        let result = tensor.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        Ok(result)
    }

    pub fn decode_batch(&self, tensors: &[Tensor]) -> Result<Vec<Vec<f32>>> {
        tensors.iter().map(|tensor| self.decode(tensor)).collect()
    }

    pub fn embedding_dim(&self) -> usize {
        384
    }
}
