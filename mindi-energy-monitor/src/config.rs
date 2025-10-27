use clap::Parser;

#[derive(Debug, Clone, Parser)]
#[command(author, version, about = "Energy monitor service", long_about = None)]
pub struct Config {
    #[arg(long, default_value_t = default_port())]
    pub port: u16,

    #[arg(long, default_value_t = default_bind_address())]
    pub bind_address: String,

    #[arg(long, default_value_t = default_collection_interval_ms())]
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
    /// Parse configuration from CLI arguments.
    pub fn parse() -> Self {
        <Self as Parser>::parse()
    }
}
