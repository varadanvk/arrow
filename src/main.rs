mod embedding;
mod vectorstore;

use anyhow::{Context, Result};
use candle_core::Device;
use clap::{Parser, Subcommand};
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType;
use std::fs;
use std::path::Path;
use uuid::Uuid;

const DEFAULT_VECTOR_STORE: &str = "vector_store.json";
const DEFAULT_MODEL: SentenceEmbeddingsModelType = SentenceEmbeddingsModelType::AllMiniLmL6V2;
const DEFAULT_CONNECTIONS: usize = 16;

/// Arrow Vector Database CLI
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Cli {
    /// Path to vector store file
    #[clap(short, long, default_value = DEFAULT_VECTOR_STORE)]
    database: String,

    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Create a new vector store
    Create {
        /// Maximum connections per node
        #[clap(short, long, default_value_t = DEFAULT_CONNECTIONS)]
        max_connections: usize,
    },

    /// Add documents to the vector store
    Add {
        /// File paths to add
        #[clap(required = true)]
        files: Vec<String>,
    },

    /// Query the vector store
    Query {
        /// The text to search for
        #[clap(required = true)]
        text: String,

        /// Number of results to return
        #[clap(short, long, default_value_t = 5)]
        top_k: usize,
    },

    /// List documents in the vector store
    List {
        /// Maximum number of documents to list
        #[clap(short, long, default_value_t = 10)]
        limit: usize,
    },

    /// Show information about the vector store
    Info,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let db_path = &cli.database;

    match cli.command {
        Commands::Create { max_connections } => create_vector_store(db_path, max_connections),
        Commands::Add { files } => add_documents(db_path, files),
        Commands::Query { text, top_k } => query_vector_store(db_path, &text, top_k),
        Commands::List { limit } => list_documents(db_path, limit),
        Commands::Info => show_info(db_path),
    }
}

fn create_vector_store(db_path: &str, max_connections: usize) -> Result<()> {
    if Path::new(db_path).exists() {
        println!("Vector store at {} already exists", db_path);
        println!("Use 'add' command to add documents to the existing store");
        return Ok(());
    }

    let store = vectorstore::VectorStore::new(Device::Cpu, max_connections);
    store.save(db_path).context("Failed to save vector store")?;

    println!("Created new vector store at {}", db_path);
    println!("Use 'add' command to add documents");

    Ok(())
}

fn add_documents(db_path: &str, files: Vec<String>) -> Result<()> {
    // Load or create the vector store
    let mut store = if Path::new(db_path).exists() {
        vectorstore::VectorStore::load(db_path, Device::Cpu)
            .context("Failed to load vector store")?
    } else {
        println!("Vector store not found, creating a new one");
        vectorstore::VectorStore::new(Device::Cpu, DEFAULT_CONNECTIONS)
    };

    // Create embedder
    let embeddor = embedding::Embeddor::new(DEFAULT_MODEL)?;

    let mut added_count = 0;
    let mut total_chunks = 0;

    for file_path in &files {
        let path = Path::new(&file_path);
        if !path.exists() {
            println!("Warning: File {} not found, skipping", file_path);
            continue;
        }

        println!("Processing file: {}", file_path);

        // Read file content
        let content = fs::read_to_string(&file_path)
            .with_context(|| format!("Failed to read file: {}", file_path))?;

        // Split into chunks
        let chunks = embeddor.chunk(&content);
        println!("  Split into {} chunks", chunks.len());
        total_chunks += chunks.len();

        // Generate embeddings
        let embeddings = embeddor.embed(&content)?;

        // Add to vector store
        for (i, (chunk, embedding)) in chunks.into_iter().zip(embeddings.into_iter()).enumerate() {
            let chunk_filename = format!("{}#chunk{}", file_path, i + 1);
            store.add_with_filename(embedding, chunk, Some(chunk_filename))?;
            added_count += 1;
        }
    }

    // Save the updated vector store
    store.save(db_path).context("Failed to save vector store")?;

    println!(
        "Successfully added {} chunks from {} files to the vector store",
        added_count,
        files.len()
    );

    Ok(())
}

fn query_vector_store(db_path: &str, query_text: &str, top_k: usize) -> Result<()> {
    if !Path::new(db_path).exists() {
        println!("Vector store not found at {}", db_path);
        println!("Use 'create' command to create a new vector store");
        return Ok(());
    }

    // Load vector store
    let store = vectorstore::VectorStore::load(db_path, Device::Cpu)
        .context("Failed to load vector store")?;

    // Create embedder
    let embeddor = embedding::Embeddor::new(DEFAULT_MODEL)?;

    // Generate query embedding
    let query_embeddings = embeddor
        .embed(query_text)
        .context("Failed to generate query embedding")?;

    if query_embeddings.is_empty() {
        println!("Query embedding could not be generated. Try a longer query.");
        return Ok(());
    }

    println!("Query: {}", query_text);

    // Use the first embedding for the query
    let query_embedding = &query_embeddings[0];
    let results = store.query(query_embedding, top_k)?;

    if results.is_empty() {
        println!("No results found.");
    } else {
        println!("\nResults:");
        for (i, (text, score, filename)) in results.iter().enumerate() {
            println!("\n  {}. Score: {:.4}", i + 1, score);
            println!("     Source: {:?}", filename);
            println!(
                "     Content: {}...",
                text.chars().take(200).collect::<String>()
            );
        }
    }

    Ok(())
}

fn list_documents(db_path: &str, limit: usize) -> Result<()> {
    if !Path::new(db_path).exists() {
        println!("Vector store not found at {}", db_path);
        println!("Use 'create' command to create a new vector store");
        return Ok(());
    }

    // Load vector store
    let store = vectorstore::VectorStore::load(db_path, Device::Cpu)
        .context("Failed to load vector store")?;

    let ids = store.get_all_ids();
    if ids.is_empty() {
        println!("Vector store is empty. Use 'add' command to add documents.");
        return Ok(());
    }

    println!(
        "Documents in the vector store (showing {} of {}):",
        std::cmp::min(limit, ids.len()),
        ids.len()
    );

    for (i, id) in ids.iter().enumerate().take(limit) {
        if let Some((text, filename)) = store.get_embedding(id) {
            println!("\n{}. ID: {}", i + 1, id);
            println!("   Source: {:?}", filename);
            println!(
                "   Content: {}...",
                text.chars().take(100).collect::<String>()
            );
        }
    }

    Ok(())
}

fn show_info(db_path: &str) -> Result<()> {
    if !Path::new(db_path).exists() {
        println!("Vector store not found at {}", db_path);
        println!("Use 'create' command to create a new vector store");
        return Ok(());
    }

    // Load vector store
    let store = vectorstore::VectorStore::load(db_path, Device::Cpu)
        .context("Failed to load vector store")?;

    println!("Vector Store Information:");
    println!("-------------------------");
    println!("Location: {}", db_path);
    println!("Document count: {}", store.text_count());

    // Get some source file stats
    let ids = store.get_all_ids();
    let mut unique_files = std::collections::HashSet::new();

    for id in &ids {
        if let Some((_, Some(filename))) = store.get_embedding(id) {
            if let Some(base_filename) = filename.split('#').next() {
                unique_files.insert(base_filename.to_string());
            }
        }
    }

    println!("Source files: {}", unique_files.len());

    if !unique_files.is_empty() {
        println!("\nSource files list:");
        for (i, file) in unique_files.iter().enumerate() {
            println!("  {}. {}", i + 1, file);
        }
    }

    Ok(())
}
