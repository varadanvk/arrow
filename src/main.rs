mod embedding;
mod vectorstore;
use anyhow::Result;
use candle_core::Device;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType; // Add this for device handling

fn main() -> Result<()> {
    // Create embedder and vector store
    let embeddor = embedding::Embeddor::new(SentenceEmbeddingsModelType::AllMiniLmL6V2)?;
    let mut store = vectorstore::VectorStore::new();

    // Example: Add a sentence embedding to the store
    let sentence = "Hello World!";
    let embeddings = embeddor.embed(sentence)?;
    for embedding in embeddings {
        store.add(embedding, sentence.to_string());
    }
    println!("Added embeddings to the store.");

    // Example: Query with another sentence
    let query = "Hello World!";
    let query_embeddings = embeddor.embed(query)?;
    println!("Querying with: {}", query);
    for query_embedding in &query_embeddings {
        let results = store.query(query_embedding)?;
        for (text, similarity) in results {
            println!("Found text: '{}' with similarity: {}", text, similarity);
        }
    }

    Ok(())
}
