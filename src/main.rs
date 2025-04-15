mod embedding;
mod vectorstore;
use anyhow::Result;
use candle_core::Device;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    // Create embedder and vector store
    let embeddor = embedding::Embeddor::new(SentenceEmbeddingsModelType::AllMiniLmL6V2)?;
    let max_connections = 16; // Maximum number of connections per node
    let mut store = vectorstore::VectorStore::new(Device::Cpu, max_connections);

    // Path for persistent storage
    let db_path = "vector_store.json";

    // Check if previously saved database exists
    if Path::new(db_path).exists() {
        println!("Loading existing vector store from {}", db_path);
        match vectorstore::VectorStore::load(db_path, Device::Cpu) {
            Ok(loaded_store) => {
                store = loaded_store;
                println!(
                    "Successfully loaded vector store with {} texts",
                    store.text_count()
                );

                // Display some sample document IDs and their filenames
                let ids = store.get_all_ids();
                if !ids.is_empty() {
                    println!("Sample documents in the store:");
                    for (i, id) in ids.iter().enumerate().take(3) {
                        if let Some((text, filename)) = store.get_embedding(id) {
                            println!(
                                "Document {}: ID={}, Filename={:?}, Content preview: {}...",
                                i + 1,
                                id,
                                filename,
                                text.chars().take(50).collect::<String>()
                            );
                        }
                    }
                }
            }
            Err(e) => {
                println!("Error loading vector store: {}. Creating a new one.", e);
            }
        }
    } else {
        // Read and process the text file
        let input_file = "test.txt";
        let text = fs::read_to_string(input_file)?;
        let chunks = embeddor.chunk(&text);
        println!("Split text into {} chunks", chunks.len());

        let embeddings = embeddor.embed(&text)?;
        println!("Generated {} embeddings", embeddings.len());

        // Add embeddings and corresponding chunks to the store
        println!("Adding embeddings to the vector store...");
        let mut added_ids = Vec::new();

        for (i, (chunk, embedding)) in chunks.into_iter().zip(embeddings.into_iter()).enumerate() {
            // Create a filename for each chunk based on the source file
            let chunk_filename = format!("{}#chunk{}", input_file, i + 1);
            let id = store.add_with_filename(embedding, chunk, Some(chunk_filename))?;
            added_ids.push(id);
        }
        println!("Finished adding {} embeddings with IDs", added_ids.len());

        // Display some of the added IDs
        if !added_ids.is_empty() {
            println!("First 3 document IDs:");
            for (i, id) in added_ids.iter().enumerate().take(3) {
                if let Some((text, filename)) = store.get_embedding(id) {
                    println!(
                        "Document {}: ID={}, Filename={:?}, Content preview: {}...",
                        i + 1,
                        id,
                        filename,
                        text.chars().take(50).collect::<String>()
                    );
                }
            }
        }

        // Save the vector store
        println!("Saving vector store to {}", db_path);
        if let Err(e) = store.save(db_path) {
            println!("Error saving vector store: {}", e);
        } else {
            println!("Vector store saved successfully");
        }
    }

    // Example: Query with a word
    let query = "A monopoly business gets stronger as it gets bigger: the fixed costs of creating a product (engineering,
management, office space) can be spread out over ever greater quantities of sales. Software startups
can enjoy especially dramatic economies of scale because the marginal cost of producing another
copy of the product is close to zero. ";
    println!("Querying with: {}", query);
    let query_embeddings = embeddor.embed(query)?;
    for query_embedding in &query_embeddings {
        let results = store.query(query_embedding, 5)?; // Get top 5 results
        println!("Query Results:");
        for (i, (text, score, filename)) in results.iter().enumerate() {
            println!("  {}. Score: {:.4}, Filename: {:?}", i + 1, score, filename);
            println!(
                "     Text: {}...",
                text.chars().take(100).collect::<String>()
            );
        }
    }

    Ok(())
}
