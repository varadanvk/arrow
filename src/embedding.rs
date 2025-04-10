use anyhow::Result;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};

pub struct Embeddor {
    model: SentenceEmbeddingsModel,
}

impl Embeddor {
    pub fn new(model_type: SentenceEmbeddingsModelType) -> Result<Self> {
        let model: SentenceEmbeddingsModel =
            SentenceEmbeddingsBuilder::remote(model_type).create_model()?;
        Ok(Self { model })
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

    pub fn embed(&self, text: &str) -> Result<Vec<Vec<f32>>> {
        let chunks = self.chunk(text);
        let embeddings = self.model.encode(&chunks)?;
        Ok(embeddings)
    }
}
