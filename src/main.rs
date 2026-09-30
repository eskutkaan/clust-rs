use clust_rs::pipeline::{run, PipelineConfig};
use std::env;
use std::path::PathBuf;
use std::process;

fn print_help() {
    eprintln!(
        "\
Clust-RS 0.2 — multi-threaded consensus clustering

USAGE:
  clust-rs <datapath> [OPTIONS]

OPTIONS:
  -o <dir>         Output directory (default: clust_rs_results)
  -n <codes...>    Normalisation codes (default: 1000)
  -K <ints...>     K values (default: 4 8 12 16 20)
  -t <float>       Tightness (default: 1.0)
  --cs <int>       Min cluster size (default: 11)
    --clusters <int> Force the exact number of output clusters
  -j <int>         Threads, 0=all CPUs (default: 0)
  --seed <int>     RNG seed (default: 42)
  --diff <float>   Binarise membership gap (default: 0.1)
  -h, --help       Show help
"
    );
}

fn main() {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        process::exit(if args.is_empty() { 1 } else { 0 });
    }

    let mut cfg = PipelineConfig::default();
    cfg.data_path = PathBuf::from(args.remove(0));

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                cfg.output_dir = PathBuf::from(&args[i]);
            }
            "-n" => {
                cfg.norm_codes.clear();
                i += 1;
                while i < args.len() && !args[i].starts_with('-') {
                    cfg.norm_codes.push(args[i].parse().unwrap_or(1000));
                    i += 1;
                }
                continue;
            }
            "-K" => {
                cfg.ks.clear();
                i += 1;
                while i < args.len() && !args[i].starts_with('-') {
                    cfg.ks.push(args[i].parse().unwrap_or(8));
                    i += 1;
                }
                continue;
            }
            "-t" => {
                i += 1;
                cfg.tightness = args[i].parse().unwrap_or(1.0);
            }
            "--cs" => {
                i += 1;
                cfg.min_cluster_size = args[i].parse().unwrap_or(11);
            }
            "--clusters" => {
                i += 1;
                let count = args[i].parse().unwrap_or(0);
                if count == 0 {
                    eprintln!("--clusters must be greater than zero");
                    process::exit(1);
                }
                cfg.forced_clusters = Some(count);
            }
            "-j" => {
                i += 1;
                cfg.jobs = args[i].parse().unwrap_or(0);
            }
            "--seed" => {
                i += 1;
                cfg.seed = args[i].parse().unwrap_or(42);
            }
            "--diff" => {
                i += 1;
                cfg.binarise_diff = args[i].parse().unwrap_or(0.1);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_help();
                process::exit(1);
            }
        }
        i += 1;
    }

    match run(&cfg) {
        Ok(res) => {
            eprintln!(
                "\nDone in {:.2}s — {} genes, {} clusters → {}",
                res.elapsed_secs,
                res.gene_ids.len(),
                res.clusters.ncols(),
                cfg.output_dir.display()
            );
        }
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    }
}
