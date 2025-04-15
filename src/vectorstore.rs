// src/vectorstore.rs
use crate::embedding::StoredEmbedding;
use candle_core::{Device, Result, Tensor};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use uuid::Uuid;

#[derive(Serialize, Deserialize)]
struct Node {
    id: Uuid,
    vector: Vec<f32>,
    neighbors: HashSet<Uuid>,
}

#[derive(Serialize, Deserialize)]
struct Layer {
    nodes: Vec<Node>,
    // Map from UUID to index in the nodes vector for fast lookup
    id_to_index: HashMap<Uuid, usize>,
}

#[derive(Serialize, Deserialize)]
pub struct VectorStore {
    layers: Vec<Layer>,
    // Map from UUID to text content
    texts: HashMap<Uuid, String>,
    // Map from UUID to filename (if applicable)
    filenames: HashMap<Uuid, String>,
    #[serde(skip)]
    #[serde(default)]
    device: Option<Device>,
    max_connections: usize,
    m_l: f32,
}

impl VectorStore {
    pub fn new(device: Device, max_connections: usize) -> Self {
        let m_l = 1.0 / (max_connections as f32).ln();
        Self {
            layers: vec![Layer {
                nodes: Vec::new(),
                id_to_index: HashMap::new(),
            }],
            texts: HashMap::new(),
            filenames: HashMap::new(),
            device: Some(device),
            max_connections,
            m_l,
        }
    }

    fn cosine_distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        let dot: f32 = v1.iter().zip(v2).map(|(a, b)| a * b).sum();
        let n1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
        1.0 - (dot / (n1 * n2)).clamp(-1.0, 1.0)
    }

    pub fn add(&mut self, embedding: Tensor, text: String) -> Result<Uuid> {
        self.add_with_filename(embedding, text, None)
    }

    pub fn add_with_filename(
        &mut self,
        embedding: Tensor,
        text: String,
        filename: Option<String>,
    ) -> Result<Uuid> {
        let vector = embedding.to_vec1::<f32>()?;
        let id = Uuid::new_v4();

        let max_level = (-rand::thread_rng().gen::<f32>().ln() * self.m_l).floor() as usize;
        while self.layers.len() <= max_level {
            self.layers.push(Layer {
                nodes: Vec::new(),
                id_to_index: HashMap::new(),
            });
        }

        for level in 0..=max_level {
            let new_node = Node {
                id: id.clone(),
                vector: vector.clone(),
                neighbors: HashSet::new(),
            };

            let node_index = self.layers[level].nodes.len();
            self.layers[level].nodes.push(new_node);
            self.layers[level].id_to_index.insert(id, node_index);

            if self.layers[level].nodes.len() > 1 {
                let nearest = self.find_nearest(&vector, level, 1)[0];
                self.connect_nodes(level, id, nearest.0);
            }
        }

        self.texts.insert(id, text);
        if let Some(fname) = filename {
            self.filenames.insert(id, fname);
        }

        Ok(id)
    }

    fn connect_nodes(&mut self, level: usize, id1: Uuid, id2: Uuid) {
        let index1 = self.layers[level].id_to_index[&id1];
        let index2 = self.layers[level].id_to_index[&id2];

        if self.layers[level].nodes[index1].neighbors.len() < self.max_connections {
            self.layers[level].nodes[index1].neighbors.insert(id2);
        }
        if self.layers[level].nodes[index2].neighbors.len() < self.max_connections {
            self.layers[level].nodes[index2].neighbors.insert(id1);
        }
    }

    pub fn query(
        &self,
        query_embedding: &Tensor,
        k: usize,
    ) -> Result<Vec<(String, f32, Option<String>)>> {
        let query = query_embedding.to_vec1::<f32>()?;

        let mut entry_point = (Uuid::nil(), f32::MAX);
        for level in (0..self.layers.len()).rev() {
            if self.layers[level].nodes.is_empty() {
                continue;
            }

            // Just get the first node as a starting point if we don't have a better one
            let first_id = self.layers[level].nodes[0].id;
            entry_point = (
                first_id,
                self.cosine_distance(&query, &self.layers[level].nodes[0].vector),
            );

            if !self.layers[level].nodes.is_empty() {
                entry_point = self.find_nearest(&query, level, 1)[0];
            }

            if level == 0 {
                break;
            }
        }

        let nearest = self.find_nearest(&query, 0, k);
        Ok(nearest
            .into_iter()
            .map(|(id, dist)| {
                let text = self.texts[&id].clone();
                let filename = self.filenames.get(&id).cloned();
                (text, 1.0 - dist, filename)
            })
            .collect())
    }

    fn find_nearest(&self, query: &[f32], level: usize, k: usize) -> Vec<(Uuid, f32)> {
        let layer = &self.layers[level];
        if layer.nodes.is_empty() {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let first_id = layer.nodes[0].id;
        let mut best = vec![(
            first_id,
            self.cosine_distance(query, &layer.nodes[0].vector),
        )];
        visited.insert(first_id);

        loop {
            let current = best[0]; // Closest unexpanded node
            let mut improved = false;

            // Check all neighbors
            let current_index = layer.id_to_index[&current.0];
            for &neighbor_id in &layer.nodes[current_index].neighbors {
                if visited.insert(neighbor_id) {
                    let neighbor_index = layer.id_to_index[&neighbor_id];
                    let dist = self.cosine_distance(query, &layer.nodes[neighbor_index].vector);
                    best.push((neighbor_id, dist));
                    best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    if best.len() > k {
                        best.pop();
                    }
                    improved = true;
                }
            }

            if !improved {
                break; // No better neighbors found
            }
        }

        best
    }

    // Serialize and save the vector store to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer(file, self)?;
        Ok(())
    }

    // Load a vector store from a file
    pub fn load<P: AsRef<Path>>(path: P, device: Device) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let mut store: VectorStore = serde_json::from_str(&contents)?;
        store.device = Some(device);
        Ok(store)
    }

    // Method to get tensor from vector for queries after loading
    pub fn vector_to_tensor(&self, vector: &[f32]) -> Result<Tensor> {
        let device = self.device.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("Device not initialized in VectorStore".to_string())
        })?;
        Tensor::from_vec(vector.to_vec(), vector.len(), device)
    }

    // Method to get the number of texts in the vector store
    pub fn text_count(&self) -> usize {
        self.texts.len()
    }

    // Get a specific embedding by ID
    pub fn get_embedding(&self, id: &Uuid) -> Option<(&String, Option<&String>)> {
        let text = self.texts.get(id)?;
        let filename = self.filenames.get(id);
        Some((text, filename))
    }

    // Get all embedding IDs
    pub fn get_all_ids(&self) -> Vec<Uuid> {
        self.texts.keys().cloned().collect()
    }

    // Add a StoredEmbedding to the vector store
    pub fn add_stored_embedding(
        &mut self,
        stored_embedding: &StoredEmbedding,
        text: String,
    ) -> Result<Uuid> {
        let device = self.device.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("Device not initialized in VectorStore".to_string())
        })?;

        let embedding = Tensor::from_vec(
            stored_embedding.vector.clone(),
            stored_embedding.vector.len(),
            device,
        )?;

        self.add_with_filename(embedding, text, Some(stored_embedding.filename.clone()))
    }
}
