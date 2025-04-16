use anyhow::Result;
use candle_core::{Device, Tensor};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};
use serde::{Deserialize, Serialize};
use std::sync::mpsc;
use std::thread;
use uuid::Uuid;

pub struct Embeddor {
    model: SentenceEmbeddingsModel,
    device: Device,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct StoredEmbedding {
    pub filename: String,
    pub id: Uuid,
    pub vector: Vec<f32>,
    #[serde(skip)]
    embedding: Option<Tensor>,
}

impl StoredEmbedding {
    pub fn new(filename: String, vector: Vec<f32>) -> Self {
        Self {
            filename,
            id: Uuid::new_v4(),
            vector,
            embedding: None,
        }
    }

    pub fn to_tensor(&mut self, device: &Device) -> Result<&Tensor> {
        if self.embedding.is_none() {
            let tensor = Tensor::from_vec(self.vector.clone(), (self.vector.len(),), device)?;
            self.embedding = Some(tensor);
        }
        Ok(self.embedding.as_ref().unwrap())
    }
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
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let num_chunks = chunks.len();
        let num_threads = std::cmp::min(4, num_chunks); // Cap at 4 threads

        if num_threads <= 1 {
            // If only one chunk or one thread, process sequentially
            let embeddings = self.model.encode(&chunks)?;
            return self.convert_to_tensors(embeddings);
        }

        // Split chunks into batches
        let chunk_size = (num_chunks + num_threads - 1) / num_threads;
        let mut chunk_batches: Vec<Vec<String>> = Vec::new();

        for i in 0..num_threads {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, num_chunks);
            if start < end {
                let batch: Vec<String> = chunks[start..end].to_vec();
                chunk_batches.push(batch);
            }
        }

        // Setup channels
        let (sender, receiver) = mpsc::channel();

        // Spawn threads
        for (thread_idx, batch) in chunk_batches.into_iter().enumerate() {
            // Clone the sender for each thread
            let thread_sender = sender.clone();

            // Move batch into thread
            thread::spawn(move || {
                // Directly call model.encode in the spawned thread
                match SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                    .create_model()
                    .and_then(|model| model.encode(&batch))
                {
                    Ok(result) => {
                        // Send successful results with thread index for ordering
                        thread_sender.send((thread_idx, Ok(result))).unwrap();
                    }
                    Err(e) => {
                        // Send error with thread index
                        thread_sender
                            .send((
                                thread_idx,
                                Err(anyhow::anyhow!("Thread {}: {}", thread_idx, e)),
                            ))
                            .unwrap();
                    }
                }
            });
        }

        // Drop the original sender to avoid deadlock
        drop(sender);

        // Collect and order results
        let mut ordered_results: Vec<(usize, Result<Vec<Vec<f32>>>)> = receiver.iter().collect();
        ordered_results.sort_by_key(|(idx, _)| *idx);

        // Process results, flatten embeddings
        let mut all_embeddings = Vec::new();
        for (_, result) in ordered_results {
            match result {
                Ok(batch_embeddings) => all_embeddings.extend(batch_embeddings),
                Err(e) => return Err(e),
            }
        }

        // Convert to tensors
        self.convert_to_tensors(all_embeddings)
    }

    fn convert_to_tensors(&self, embeddings: Vec<Vec<f32>>) -> Result<Vec<Tensor>> {
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
