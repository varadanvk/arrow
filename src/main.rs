mod embedding;
mod vectorstore;
use anyhow::Result;
use candle_core::Device;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType;

fn main() -> Result<()> {
    // Create embedder and vector store
    let embeddor = embedding::Embeddor::new(SentenceEmbeddingsModelType::AllMiniLmL6V2)?;
    let max_connections = 16; // Maximum number of connections per node
    let mut store = vectorstore::VectorStore::new(Device::Cpu, max_connections);

    // Example: Add a sentence embedding to the store
    let sentence = "Hello World!";
    let embeddings = embeddor.embed(sentence)?;
    for embedding in embeddings {
        store.add(embedding, sentence.to_string())?;
    }
    println!("Added embeddings to the store.");

    // Example: Query with another sentence
    let query = "Hello World!";
    let query_embeddings = embeddor.embed(query)?;
    println!("Querying with: {}", query);
    for query_embedding in &query_embeddings {
        let results = store.query(query_embedding, 5)?; // Get top 5 results
        for (text, similarity) in results {
            println!("Found text: '{}' with similarity: {}", text, similarity);
        }
    }

    Ok(())
}
