// src/vectorstore.rs
use candle_core::{Device, Result, Tensor};
use rand::Rng;
use std::collections::{HashMap, HashSet};

struct Node {
    vector: Vec<f32>,
    neighbors: HashSet<usize>,
}

struct Layer {
    nodes: Vec<Node>,
}

pub struct VectorStore {
    layers: Vec<Layer>,
    texts: HashMap<usize, String>,
    device: Device,
    max_connections: usize,
    m_l: f32,
}

impl VectorStore {
    pub fn new(device: Device, max_connections: usize) -> Self {
        let m_l = 1.0 / (max_connections as f32).ln();
        Self {
            layers: vec![Layer { nodes: Vec::new() }],
            texts: HashMap::new(),
            device,
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

    pub fn add(&mut self, embedding: Tensor, text: String) -> Result<()> {
        let vector = embedding.to_vec1::<f32>()?;
        let id = self.layers[0].nodes.len();

        let max_level = (-rand::thread_rng().gen::<f32>().ln() * self.m_l).floor() as usize;
        while self.layers.len() <= max_level {
            self.layers.push(Layer { nodes: Vec::new() });
        }

        for level in 0..=max_level {
            let new_node = Node {
                vector: vector.clone(),
                neighbors: HashSet::new(),
            };
            let layer_id = if level == 0 {
                id
            } else {
                self.layers[level].nodes.len()
            };
            self.layers[level].nodes.push(new_node);

            if !self.layers[level].nodes.is_empty() {
                let nearest = self.find_nearest(&vector, level, 1)[0];
                self.connect_nodes(level, layer_id, nearest.0);
            }
        }

        self.texts.insert(id, text);
        Ok(())
    }

    fn connect_nodes(&mut self, level: usize, id1: usize, id2: usize) {
        let layer = &mut self.layers[level];
        if layer.nodes[id1].neighbors.len() < self.max_connections {
            layer.nodes[id1].neighbors.insert(id2);
        }
        if layer.nodes[id2].neighbors.len() < self.max_connections {
            layer.nodes[id2].neighbors.insert(id1);
        }
    }

    pub fn query(&self, query_embedding: &Tensor, k: usize) -> Result<Vec<(String, f32)>> {
        let query = query_embedding.to_vec1::<f32>()?;

        let mut entry_point = (0, f32::MAX);
        for level in (0..self.layers.len()).rev() {
            if self.layers[level].nodes.is_empty() {
                continue;
            }
            entry_point = self.find_nearest(&query, level, 1)[0];
            if level == 0 {
                break;
            }
        }

        let nearest = self.find_nearest(&query, 0, k);
        Ok(nearest
            .into_iter()
            .map(|(id, dist)| (self.texts[&id].clone(), 1.0 - dist))
            .collect())
    }

    fn find_nearest(&self, query: &[f32], level: usize, k: usize) -> Vec<(usize, f32)> {
        let layer = &self.layers[level];
        if layer.nodes.is_empty() {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut best = vec![(0, self.cosine_distance(query, &layer.nodes[0].vector))];
        visited.insert(0);

        loop {
            let current = best[0]; // Closest unexpanded node
            let mut improved = false;

            // Check all neighbors
            for &neighbor in &layer.nodes[current.0].neighbors {
                if visited.insert(neighbor) {
                    let dist = self.cosine_distance(query, &layer.nodes[neighbor].vector);
                    best.push((neighbor, dist));
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
}
