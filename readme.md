# Arrow

A mini vector database built in rust

What is a vector database: https://www.pinecone.io/learn/vector-database/

Project structure:

- `vectorstore.rs`: The actual vector database.
  - Start with a list of vectors

Note: Worry about this later

- `embedding.rs`: to take in the data and turn it into a vector embedding. Also supports chunking functionality for every token.
  - will use huggingface tokenizers for now
  - [huggingface rust](https://www.shuttle.dev/blog/2024/05/01/using-huggingface-rust)
