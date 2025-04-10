// mod vectorstore;
// use candle_core::{DType, Device, Tensor};
// use candle_nn::Embedding;
// use vectorstore::VectorStore;

// fn main() -> anyhow::Result<()> {
//     let device = Device::Cpu;
//     let mut store = VectorStore::new();
//     let sentence = "Hello world";
//     let embeddor = Embeddor::new("all-minilm-l6-v2")?;
//     let embedding = embeddor.embed(sentence)?;
//     println!("{:?}", embedding);
//     Ok(())
// }
mod embedding;
use anyhow::Result;
use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};
fn main() -> anyhow::Result<()> {
    let embeddor = embedding::Embeddor::new(SentenceEmbeddingsModelType::AllMiniLmL6V2)?;

    let input = "Rust is a multi-paradigm, general-purpose programming language. \
        Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety—that is, \
        that all references point to valid memory—without requiring the use of a garbage collector or \
        reference counting present in other memory-safe languages. To simultaneously enforce \
        memory safety and prevent concurrent data races, Rust's borrow checker tracks the object lifetime \
        and variable scope of all references in a program during compilation. Rust is popular for \
        systems programming but also offers high-level features including functional programming constructs.";
    let embeddigns = embeddor.embed(input)?;
    for embedding in embeddigns {
        println!("{:?}", embedding);
    }

    Ok(())
}
