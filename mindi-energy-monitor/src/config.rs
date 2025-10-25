use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default = "default_bind_address")]
    pub bind_address: String,

    #[serde(default = "default_collection_interval_ms")]
    pub collection_interval_ms: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: default_port(),
            bind_address: default_bind_address(),
            collection_interval_ms: default_collection_interval_ms(),
        }
    }
}

fn default_port() -> u16 {
    50053
}

fn default_bind_address() -> String {
    "127.0.0.1".to_string()
}

fn default_collection_interval_ms() -> u64 {
    50
}

impl Config {
    /// Load configuration from file
    pub fn from_file(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Load configuration from file or use defaults
    pub fn load() -> Self {
        let config_path = std::path::Path::new("mindi-energy-monitor.toml");
        if config_path.exists() {
            match Self::from_file(config_path) {
                Ok(config) => config,
                Err(e) => {
                    tracing::warn!("Failed to load config file: {}. Using defaults.", e);
                    Self::default()
                }
            }
        } else {
            Self::default()
        }
    }
}
