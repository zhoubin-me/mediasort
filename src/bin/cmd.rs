use anyhow::{Context, Result};
use clap::Parser;
use home::home_dir;
use mediasort::utils::*;
use std::collections::HashMap;
use std::{fs, path::Path};
fn main() -> Result<()> {
    let cli = Cli::parse();
    let source_dir = Path::new(&cli.source);
    if !source_dir.exists() {
        anyhow::bail!("Source directory does not exist: {}", cli.source);
    }
    let (media_type, cache_path) = if cli.video {
        (&Media::Video, home_dir().unwrap().join(".video_cache"))
    } else {
        (&Media::Photo, home_dir().unwrap().join(".photo_cache"))
    };

    // Try to load photo_infos from cache file
    let mut cached_photos = load_cached_photos(&cache_path);

    // Extract photos from source directory
    let photo_infos: HashMap<String, PhotoInfo> =
        extract_photos(source_dir, media_type, &cached_photos);
    // Update cache with new photo information
    cached_photos.extend(photo_infos.clone());
    // Save photo_infos to cache file
    save_cached_photos(&cache_path, &cached_photos)?;

    // Print statistics
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n📸 Media Collection Statistics 📸");
    print_stats(&cached_photos)?;

    // Remove duplicates
    remove_duplicates(&mut cached_photos);

    // Print statistics after removing duplicates
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n📸 Media Collection Statistics after Removing Duplicates 📸");
    print_stats(&cached_photos)?;

    // Organize photos to target directory
    println!("Do you want to organize media to target directory? [y/N]");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let input = input.trim().to_lowercase();
    if input != "y" && input != "yes" {
        println!("Skipping photo organization.");
        return Ok(());
    }
    println!("📸 Organizing media...");
    let target_dir = Path::new(&cli.target);
    if target_dir.exists() {
        println!(
            "Target directory already exists: {}\n Skipping media organization.",
            cli.target
        );
    } else {
        organize_photos(&target_dir, &cached_photos)?;
    }

    if !cli.video {
        // Similar photos search and re-organize
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        similarity_search(&cli.target)?;
    }
    return Ok(());
}
