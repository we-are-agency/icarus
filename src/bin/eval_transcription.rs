use icarus::transcription::{
    build_standardized_evaluation_report, format_evaluation_report, summarize_evaluation_report,
};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dataset_dir = PathBuf::from("midi");
    let mut json_out: Option<PathBuf> = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => {
                let Some(value) = args.next() else {
                    return Err("--dataset requires a path".into());
                };
                dataset_dir = PathBuf::from(value);
            }
            "--json" => {
                let Some(value) = args.next() else {
                    return Err("--json requires a path".into());
                };
                json_out = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                eprintln!("Usage: cargo run --bin eval_transcription -- [--dataset DIR] [--json PATH]");
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let report = build_standardized_evaluation_report(&dataset_dir)?;
    print!("{}", format_evaluation_report(&report));

    if let Some(path) = json_out {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let summary = summarize_evaluation_report(&report);
        fs::write(&path, serde_json::to_string(&summary)?)?;
        eprintln!("wrote JSON report to {}", path.display());
    }

    Ok(())
}
