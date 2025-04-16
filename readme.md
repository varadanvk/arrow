# Arrow

A lightweight vector database built in Rust with persistent storage.

## Features

- Efficient vector similarity search
- Persistent JSON storage
- UUID-based document identification
- Support for associating vectors with filenames
- Simple CLI interface for common operations

## Installation

1. Clone the repository:

```bash
git clone https://github.com/varadanvk/arrow.git
cd arrow
```

2. Build the project:

```bash
cargo build --release
```

3. Run the executable:

```bash
./target/release/arrow
```

## CLI Usage

The CLI provides several commands to interact with the vector database:

### Global Options

- `-d, --database <PATH>`: Specify the path to the vector store file (default: `vector_store.json`)
- `-h, --help`: Print help information
- `-V, --version`: Print version information

### Commands

#### Create a new vector store

```bash
arrow create [OPTIONS]
```

Options:

- `-m, --max-connections <NUM>`: Maximum connections per node (default: 16)

Example:

```bash
arrow create --max-connections 32
```

#### Add documents to the vector store

```bash
arrow add <FILES>...
```

Example:

```bash
arrow add document1.txt document2.txt
```

This will:

1. Read the text from each file
2. Split it into chunks (max 512 characters each)
3. Generate embeddings using the All-MiniLM-L6-v2 model
4. Add each chunk with its embedding to the vector store
5. Save the updated vector store to disk

#### Query the vector store

```bash
arrow query [OPTIONS] <TEXT>
```

Options:

- `-t, --top-k <NUM>`: Number of results to return (default: 5)

Example:

```bash
arrow query "What is a monopoly business?" --top-k 3
```

#### List documents in the vector store

```bash
arrow list [OPTIONS]
```

Options:

- `-l, --limit <NUM>`: Maximum number of documents to list (default: 10)

Example:

```bash
arrow list --limit 20
```

#### Show vector store information

```bash
arrow info
```

This displays:

- The location of the vector store
- The number of documents
- The source files

## Architecture

Arrow consists of two main components:

1. **VectorStore**: A hierarchical navigable small-world (HNSW) graph-based vector index with:

   - Multiple layers for efficient navigation
   - Configurable maximum connections per node
   - UUID-based document identification

2. **Embeddor**: A text embedding module that:
   - Uses Hugging Face's Rust implementation of All-MiniLM-L6-v2
   - Supports chunking of long texts
   - Processes embeddings in parallel for better performance

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
