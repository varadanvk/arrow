mod embedding;
mod vectorstore;

use anyhow::{Context, Result};
use candle_core::Device;
use clap::{Parser, Subcommand};
use colored::*;
use console::Term;
use indicatif::{ProgressBar, ProgressStyle};
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType;
use std::fs;
use std::path::Path;
use tabled::settings::Style;
use tabled::{Table, Tabled};

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
    let term = Term::stdout();
    if Path::new(db_path).exists() {
        term.write_line(&format!(
            "{}",
            "Vector store already exists".yellow().bold()
        ))?;
        term.write_line(&format!("  Path: {}", db_path.bright_blue()))?;
        term.write_line(&format!(
            "{}",
            "Use 'add' command to add documents to the existing store".italic()
        ))?;
        return Ok(());
    }

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
            .template("{spinner:.green} {msg}")?,
    );
    spinner.set_message(format!(
        "Creating vector store with {} max connections...",
        max_connections
    ));
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    let store = vectorstore::VectorStore::new(Device::Cpu, max_connections);
    store.save(db_path).context("Failed to save vector store")?;

    spinner.finish_with_message(format!(
        "{}✓{} Vector store created successfully!",
        "[".green(),
        "]".green()
    ));
    term.write_line("")?;
    term.write_line(&format!("  {} {}", "Location:".blue(), db_path))?;
    term.write_line(&format!(
        "  {}",
        "Use 'add' command to add documents".italic()
    ))?;

    Ok(())
}

fn add_documents(db_path: &str, files: Vec<String>) -> Result<()> {
    let term = Term::stdout();
    term.write_line(&format!(
        "{}",
        "Arrow Vector Database".bright_green().bold()
    ))?;
    term.write_line("")?;

    // Load or create the vector store
    let load_spinner = ProgressBar::new_spinner();
    load_spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
            .template("{spinner:.blue} {msg}")?,
    );
    load_spinner.enable_steady_tick(std::time::Duration::from_millis(100));
    load_spinner.set_message("Loading vector store...");

    let mut store = if Path::new(db_path).exists() {
        let store = vectorstore::VectorStore::load(db_path, Device::Cpu)
            .context("Failed to load vector store")?;
        load_spinner.finish_with_message(format!(
            "{}✓{} Vector store loaded from {}",
            "[".green(),
            "]".green(),
            db_path.bright_blue()
        ));
        store
    } else {
        load_spinner.finish_with_message(format!(
            "{}!{} Vector store not found, creating a new one",
            "[".yellow(),
            "]".yellow()
        ));
        vectorstore::VectorStore::new(Device::Cpu, DEFAULT_CONNECTIONS)
    };

    // Create embedder
    term.write_line("")?;
    let embed_spinner = ProgressBar::new_spinner();
    embed_spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
            .template("{spinner:.magenta} {msg}")?,
    );
    embed_spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    let embeddor = embedding::Embeddor::new(DEFAULT_MODEL)?;
    embed_spinner.finish_with_message(format!("{}✓{} Embedding model initialized", "[".green(), "]".green()));

    let mut added_count = 0;
    let mut _total_chunks = 0;
    let mut processed_files = 0;

    // Create multi-file progress bar
    let files_progress = ProgressBar::new(files.len() as u64);
    files_progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len} files processed")?
            .progress_chars("█▓▒░ "),
    );

    for file_path in &files {
        let path = Path::new(&file_path);
        if !path.exists() {
            term.write_line(&format!(
                "{} File not found: {}",
                "[WARNING]".yellow().bold(),
                file_path
            ))?;
            files_progress.inc(1);
            continue;
        }

        term.write_line(&format!(
            "\n{} {}",
            "Processing file:".blue().bold(),
            file_path.bright_white()
        ))?;

        // Read file content
        let content = fs::read_to_string(&file_path)
            .with_context(|| format!("Failed to read file: {}", file_path))?;

        // Split into chunks
        let chunks = embeddor.chunk(&content);
        term.write_line(&format!(
            "  Split into {} chunks",
            chunks.len().to_string().cyan()
        ))?;
        _total_chunks += chunks.len();

        // Generate embeddings
        let embedding_progress = ProgressBar::new(chunks.len() as u64);
        embedding_progress.set_style(
            ProgressStyle::default_bar()
                .template(
                    "  Generating embeddings: [{elapsed_precise}] {bar:.green} {pos}/{len} chunks",
                )?
                .progress_chars("█▓▒░ "),
        );

        let embeddings = embeddor.embed(&content)?;
        embedding_progress.finish_and_clear();

        // Add to vector store with progress
        let store_progress = ProgressBar::new(chunks.len() as u64);
        store_progress.set_style(ProgressStyle::default_bar()
            .template("  Adding to vector store: [{elapsed_precise}] {bar:.yellow} {pos}/{len} chunks")?
            .progress_chars("█▓▒░ "));

        for (i, (chunk, embedding)) in chunks.into_iter().zip(embeddings.into_iter()).enumerate() {
            let chunk_filename = format!("{}#chunk{}", file_path, i + 1);
            store.add_with_filename(embedding, chunk, Some(chunk_filename))?;
            added_count += 1;
            store_progress.inc(1);
        }
        store_progress.finish_and_clear();
        processed_files += 1;
        files_progress.inc(1);
    }
    files_progress.finish();

    // Save the updated vector store
    let save_spinner = ProgressBar::new_spinner();
    save_spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
            .template("{spinner:.green} {msg}")?,
    );
    save_spinner.enable_steady_tick(std::time::Duration::from_millis(100));
    save_spinner.set_message("Saving vector store...");

    store.save(db_path).context("Failed to save vector store")?;
    save_spinner.finish_with_message(format!(
        "{}✓{} Vector store saved to {}",
        "[".green(),
        "]".green(),
        db_path.bright_blue()
    ));

    term.write_line("")?;
    term.write_line(&format!("{}", "Summary:".bold().underline()))?;
    term.write_line(&format!(
        "  {} {} {}",
        "Added".green(),
        added_count.to_string().bright_white(),
        "chunks"
    ))?;
    term.write_line(&format!(
        "  {} {} {}",
        "From".green(),
        processed_files.to_string().bright_white(),
        "files"
    ))?;
    term.write_line(&format!("  {} {}", "Database:".green(), db_path))?;
    
    // Add chunks progress bar visualization
    let chunk_bar_width = 40;
    let chunk_pct = if _total_chunks > 0 { (added_count as f32) / (_total_chunks as f32) } else { 1.0 };
    let filled = (chunk_pct * chunk_bar_width as f32) as usize;
    let empty = chunk_bar_width - filled;
    
    term.write_line("")?;
    term.write_line(&format!("{}", "Chunks processed:".blue().bold()))?;
    term.write_line(&format!("[{}{}] {:.1}%", 
        "█".repeat(filled).bright_green(),
        "▒".repeat(empty).bright_black(),
        chunk_pct * 100.0))?;

    Ok(())
}

#[derive(Tabled)]
struct QueryResult {
    #[tabled(rename = "#")]
    index: usize,
    #[tabled(rename = "Score")]
    score: String,
    #[tabled(rename = "Source")]
    source: String,
    #[tabled(rename = "Content")]
    content: String,
}

fn query_vector_store(db_path: &str, query_text: &str, top_k: usize) -> Result<()> {
    let term = Term::stdout();
    if !Path::new(db_path).exists() {
        term.write_line(&format!("{}", "Vector store not found".red().bold()))?;
        term.write_line(&format!("  Expected at: {}", db_path))?;
        term.write_line(&format!(
            "{}",
            "Use 'create' command to create a new vector store".italic()
        ))?;
        return Ok(());
    }

    // Show banner
    term.write_line(&format!("{}", "Arrow Vector Search".bright_green().bold()))?;
    term.write_line("")?;

    // Load vector store
    let load_spinner = ProgressBar::new_spinner();
    load_spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
            .template("{spinner:.blue} {msg}")?,
    );
    load_spinner.enable_steady_tick(std::time::Duration::from_millis(100));
    load_spinner.set_message("Loading vector store...");

    let store = vectorstore::VectorStore::load(db_path, Device::Cpu)
        .context("Failed to load vector store")?;
    load_spinner.finish_with_message(format!(
        "{}✓{} Vector store loaded",
        "[".green(),
        "]".green()
    ));

    // Create embedder
    let embed_spinner = ProgressBar::new_spinner();
    embed_spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
            .template("{spinner:.magenta} {msg}")?,
    );
    embed_spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    let embeddor = embedding::Embeddor::new(DEFAULT_MODEL)?;
    embed_spinner.finish_with_message(format!(
        "{}✓{} Embedding model ready",
        "[".green(),
        "]".green()
    ));

    // Generate query embedding
    let query_spinner = ProgressBar::new_spinner();
    query_spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
            .template("{spinner:.yellow} {msg}")?,
    );
    query_spinner.enable_steady_tick(std::time::Duration::from_millis(100));
    query_spinner.set_message("Generating query embedding...");

    let query_embeddings = embeddor
        .embed(query_text)
        .context("Failed to generate query embedding")?;

    if query_embeddings.is_empty() {
        query_spinner.finish_with_message(format!(
            "{}✗{} Failed to generate embedding",
            "[".red(),
            "]".red()
        ));
        term.write_line(&format!(
            "{}",
            "Query embedding could not be generated. Try a longer query.".yellow()
        ))?;
        return Ok(());
    }
    query_spinner.finish_with_message(format!(
        "{}✓{} Query embedding generated",
        "[".green(),
        "]".green()
    ));

    // Display query
    term.write_line("")?;
    term.write_line(&format!(
        "{} {}",
        "Query:".blue().bold(),
        query_text.bright_white()
    ))?;

    // Use the first embedding for the query
    let search_spinner = ProgressBar::new_spinner();
    search_spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
            .template("{spinner:.cyan} {msg}")?,
    );
    search_spinner.enable_steady_tick(std::time::Duration::from_millis(100));
    search_spinner.set_message(format!("Searching for top {} matches...", top_k));

    let query_embedding = &query_embeddings[0];
    let results = store.query(query_embedding, top_k)?;
    search_spinner.finish_with_message(format!("{}✓{} Search complete", "[".green(), "]".green()));

    if results.is_empty() {
        term.write_line(&format!("{}", "\nNo results found.".yellow().bold()))?;
    } else {
        term.write_line(&format!("{}", "\nResults:".green().bold()))?;

        let table_results = results
            .iter()
            .enumerate()
            .map(|(i, (text, score, filename))| QueryResult {
                index: i + 1,
                score: format!("{:.4}", score),
                source: match filename {
                    Some(f) => f.clone(),
                    None => "Unknown".to_string(),
                },
                content: text.chars().take(100).collect::<String>() + "...",
            })
            .collect::<Vec<_>>();

        let mut binding = Table::new(table_results);
        let table = binding.with(Style::modern().to_owned());

        term.write_line(&format!("{}", table))?;
    }

    Ok(())
}

#[derive(Tabled)]
struct Document {
    #[tabled(rename = "#")]
    index: usize,
    #[tabled(rename = "ID")]
    id: String,
    #[tabled(rename = "Source")]
    source: String,
    #[tabled(rename = "Preview")]
    preview: String,
}

fn list_documents(db_path: &str, limit: usize) -> Result<()> {
    let term = Term::stdout();
    if !Path::new(db_path).exists() {
        term.write_line(&format!("{}", "Vector store not found".red().bold()))?;
        term.write_line(&format!("  Expected at: {}", db_path))?;
        term.write_line(&format!(
            "{}",
            "Use 'create' command to create a new vector store".italic()
        ))?;
        return Ok(());
    }

    // Show banner
    term.write_line(&format!(
        "{}",
        "Arrow Vector Database".bright_green().bold()
    ))?;
    term.write_line("")?;

    // Load vector store
    let load_spinner = ProgressBar::new_spinner();
    load_spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
            .template("{spinner:.blue} {msg}")?,
    );
    load_spinner.enable_steady_tick(std::time::Duration::from_millis(100));
    load_spinner.set_message("Loading vector store...");

    let store = vectorstore::VectorStore::load(db_path, Device::Cpu)
        .context("Failed to load vector store")?;
    load_spinner.finish_with_message(format!(
        "{}✓{} Vector store loaded",
        "[".green(),
        "]".green()
    ));

    let ids = store.get_all_ids();
    if ids.is_empty() {
        term.write_line(&format!("{}", "\nVector store is empty".yellow().bold()))?;
        term.write_line(&format!(
            "{}",
            "Use 'add' command to add documents".italic()
        ))?;
        return Ok(());
    }

    term.write_line("")?;
    term.write_line(&format!(
        "{} {} {} {}",
        "Documents in the vector store".blue().bold(),
        format!(
            "(showing {} of {})",
            std::cmp::min(limit, ids.len()),
            ids.len()
        )
        .bright_black(),
        "db:".blue().bold(),
        db_path.bright_white()
    ))?;

    let mut documents = Vec::new();

    for (i, id) in ids.iter().enumerate().take(limit) {
        if let Some((text, filename)) = store.get_embedding(id) {
            documents.push(Document {
                index: i + 1,
                id: id.to_string().chars().take(8).collect::<String>() + "...",
                source: match filename {
                    Some(f) => f.clone(),
                    None => "Unknown".to_string(),
                },
                preview: text.chars().take(60).collect::<String>() + "...",
            });
        }
    }

    let mut binding = Table::new(documents);
    let table = binding.with(Style::modern().to_owned());

    term.write_line(&format!("{}", table))?;

    Ok(())
}

#[derive(Tabled)]
struct SourceFile {
    #[tabled(rename = "#")]
    index: usize,
    #[tabled(rename = "File")]
    filename: String,
}

fn show_info(db_path: &str) -> Result<()> {
    let term = Term::stdout();
    if !Path::new(db_path).exists() {
        term.write_line(&format!("{}", "Vector store not found".red().bold()))?;
        term.write_line(&format!("  Expected at: {}", db_path))?;
        term.write_line(&format!(
            "{}",
            "Use 'create' command to create a new vector store".italic()
        ))?;
        return Ok(());
    }

    // Show banner
    term.write_line(&format!(
        "{}",
        "Arrow Vector Database".bright_green().bold()
    ))?;
    term.write_line("")?;

    // Load vector store
    let load_spinner = ProgressBar::new_spinner();
    load_spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
            .template("{spinner:.blue} {msg}")?,
    );
    load_spinner.enable_steady_tick(std::time::Duration::from_millis(100));
    load_spinner.set_message("Loading vector store...");

    let store = vectorstore::VectorStore::load(db_path, Device::Cpu)
        .context("Failed to load vector store")?;
    load_spinner.finish_with_message(format!(
        "{}✓{} Vector store loaded",
        "[".green(),
        "]".green()
    ));

    // Display info box
    term.write_line("")?;
    let border = "╔═══════════════════════════════════════════╗".bright_blue();
    let bottom_border = "╚═══════════════════════════════════════════╝".bright_blue();
    term.write_line(&format!("{}", border))?;
    term.write_line(&format!(
        "{} {:<40} {}",
        "║".bright_blue(),
        format!(
            "  {} {}",
            "Vector Store Information".bright_white().bold(),
            ""
        ),
        "║".bright_blue()
    ))?;
    term.write_line(&format!(
        "{} {:<40} {}",
        "╠".bright_blue(),
        "═══════════════════════════════════════════".bright_blue(),
        "╣".bright_blue()
    ))?;
    term.write_line(&format!(
        "{} {:<40} {}",
        "║".bright_blue(),
        "",
        "║".bright_blue()
    ))?;
    term.write_line(&format!(
        "{} {:<40} {}",
        "║".bright_blue(),
        format!("  {}: {}", "Location".green(), db_path),
        "║".bright_blue()
    ))?;
    term.write_line(&format!(
        "{} {:<40} {}",
        "║".bright_blue(),
        format!(
            "  {}: {}",
            "Document count".green(),
            store.text_count().to_string().bright_white()
        ),
        "║".bright_blue()
    ))?;

    // Get source file stats
    let ids = store.get_all_ids();
    let mut unique_files = std::collections::HashSet::new();

    for id in &ids {
        if let Some((_, Some(filename))) = store.get_embedding(id) {
            if let Some(base_filename) = filename.split('#').next() {
                unique_files.insert(base_filename.to_string());
            }
        }
    }

    term.write_line(&format!(
        "{} {:<40} {}",
        "║".bright_blue(),
        format!(
            "  {}: {}",
            "Source files".green(),
            unique_files.len().to_string().bright_white()
        ),
        "║".bright_blue()
    ))?;
    term.write_line(&format!(
        "{} {:<40} {}",
        "║".bright_blue(),
        "",
        "║".bright_blue()
    ))?;
    term.write_line(&format!("{}", bottom_border))?;

    if !unique_files.is_empty() {
        term.write_line("")?;
        term.write_line(&format!("{}", "Source Files:".blue().bold()))?;

        let source_files = unique_files
            .iter()
            .enumerate()
            .map(|(i, filename)| SourceFile {
                index: i + 1,
                filename: filename.to_string(),
            })
            .collect::<Vec<_>>();

        let mut binding = Table::new(source_files);
        let table = binding.with(Style::psql().to_owned());

        term.write_line(&format!("{}", table))?;
    }

    Ok(())
}
