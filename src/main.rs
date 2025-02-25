use anyhow::{Context, Result};
use blake3;
use chrono::{Datelike, NaiveDateTime};
use clap::Parser;
use console::style;
use exif::Reader;
use indicatif::{ProgressBar, ProgressStyle};
use libc::dlopen;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use tch::vision::dinov2::{vit_base, vit_giant, vit_large, vit_small};
use tch::vision::imagenet;
use walkdir::WalkDir;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Source directory path
    #[arg(short, long)]
    source: String,

    /// Target directory path
    #[arg(short, long)]
    target: String,

    /// Whether to sort video files
    #[arg(short, long, default_value_t = false)]
    video: bool,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
struct PhotoInfo {
    path: PathBuf,
    hash: String,
    date: NaiveDateTime,
    size: u64,
}

#[derive(Debug, PartialEq)]
pub enum Media {
    Photo,
    Video,
    Other,
}

const SIMILARITY_THRESHOLD: f32 = 0.999;

fn main() -> Result<()> {
    let cli = Cli::parse();
    let source_dir = Path::new(&cli.source);
    if !source_dir.exists() {
        anyhow::bail!("Source directory does not exist: {}", cli.source);
    }
    let (media_type, cache_file) = if cli.video {
        (&Media::Video, ".video_cache")
    } else {
        (&Media::Photo, ".photo_cache")
    };


    // Try to load photo_infos from cache file
    let cache_path = Path::new(cache_file);
    let mut cached_photos = load_cached_photos(cache_path);

    // Extract photos from source directory
    let photo_infos: HashMap<String, PhotoInfo> = extract_photos(source_dir, media_type, &cached_photos);
    // Update cache with new photo information
    cached_photos.extend(photo_infos.clone());
    // Save photo_infos to cache file
    let cache_path = Path::new(cache_file);
    let cache_file = fs::File::create(cache_path).with_context(|| "Failed to create cache file")?;
    serde_json::to_writer(cache_file, &cached_photos)
        .with_context(|| "Failed to write cache file")?;


    // Print statistics
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n📸 Photo Collection Statistics 📸");
    print_stats(&cached_photos)?;


    // Remove duplicates
    remove_duplicates(&mut cached_photos);

    // Print statistics after removing duplicates
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n📸 Photo Collection Statistics after Removing Duplicates 📸");
    print_stats(&cached_photos)?;


    // Organize photos to target directory
    println!("Do you want to organize photos to target directory? [y/N]");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let input = input.trim().to_lowercase();
    if input != "y" && input != "yes" {
        println!("Skipping photo organization.");
        return Ok(());
    }
    println!("📸 Organizing photos...");
    let target_dir = Path::new(&cli.target);
    if target_dir.exists() {
        println!(
            "Target directory already exists: {}\n Skipping photo organization.",
            cli.target
        );
    } else {
        organize_photos(&target_dir, &cached_photos)?;
    }

    // Similar photos search and re-organize
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    similarity_search(&cli.target)?;
    return Ok(());
}


fn extract_photos(source_dir: &Path, media_type: &Media, cached_photos: &HashMap<String, PhotoInfo>) -> HashMap<String, PhotoInfo> {
    println!("📸 Collecting photos...");
    let pb = ProgressBar::new(collect_photos(source_dir, &media_type).count() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    collect_photos(source_dir, media_type)
        .par_bridge()
        .filter_map(|path| {
            pb.inc(1);
            let path_str = path.display().to_string();
            if let Some(cached_info) = cached_photos.get(&path_str) {
                // Check if file size and modified time match to validate cache
                if let Ok(metadata) = fs::metadata(&path) {
                    if metadata.len() == cached_info.size {
                        return Some((path_str, cached_info.clone()));
                    }
                }
            }
            let hash = calculate_hash(&path).unwrap();
            let exif = get_exif_info(&path);
            let date = get_date_from_exif(&exif)?;
            let size = fs::metadata(&path).unwrap().len();
            Some((
                path.display().to_string(),
                PhotoInfo {
                    path,
                    hash,
                    date,
                    size,
                },
            ))
        })
        .collect()
}

fn load_cached_photos(cache_path: &Path) -> HashMap<String, PhotoInfo> {
    match fs::File::open(cache_path) {
        Ok(cache_file) => {
            match serde_json::from_reader::<_, HashMap<String, PhotoInfo>>(cache_file) {
                Ok(cached_photos) => {
                    println!("📸 Loading cached photo information...");
                    cached_photos
                }
                Err(_) => {
                    println!("⚠️  Cache file is corrupted, starting fresh");
                    HashMap::new()
                }
            }
        }
        Err(_) => {
            println!("⚠️ No cache file found, starting fresh");
            HashMap::new()
        }
    }
}

fn organize_photos(target_dir: &Path, cached_photos: &HashMap<String, PhotoInfo>) -> Result<()> {
    let pb = ProgressBar::new(cached_photos.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );
    cached_photos.values().par_bridge().try_for_each(|photo| {
        // Create YYYY-MM directory structure
        let year = photo.date.year();
        let month = photo.date.month();
        let target_subdir = target_dir.join(format!("{:04}-{:02}", year, month));
        if let Ok(metadata) = fs::metadata(&photo.path) {
            if metadata.len() < 10240 && photo.date == NaiveDateTime::UNIX_EPOCH {
                return Ok::<(), anyhow::Error>(());
            }
        }

        if let Some(ext) = photo.path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            if ext == "ico" || ext == "bmp" {
                return Ok::<(), anyhow::Error>(());
            }
        }

        // Create directory if it doesn't exist
        fs::create_dir_all(&target_subdir)
            .with_context(|| format!("Failed to create directory: {}", target_subdir.display()))?;

        // Get filename from original path
        let filename = photo
            .path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid filename"))?;

        // Copy file if target doesn't exist
        let mut target_path = target_subdir.join(filename);
        let mut counter = 1;

        while target_path.exists() {
            // Get file stem and extension
            // Split filename into stem and extension
            let filename_str = filename.to_string_lossy();
            let (stem, ext) = match filename_str.rsplit_once('.') {
                Some((s, e)) => (s.to_string(), e.to_string()),
                None => (filename_str.to_string(), String::new()),
            };

            // Create new filename with counter
            let new_filename = if ext.is_empty() {
                format!("{}_{}", stem, counter)
            } else {
                format!("{}_{}.{}", stem, counter, ext)
            };
            target_path = target_subdir.join(new_filename);
            counter += 1;
        }

        fs::copy(&photo.path, &target_path).with_context(|| {
            format!(
                "Failed to copy {} to {}",
                photo.path.display(),
                target_path.display()
            )
        })?;

        pb.inc(1);
        Ok::<(), anyhow::Error>(())
    })?;

    pb.finish();
    println!("\n✨ Photos organized successfully in target directory");
    Ok(())
}

fn similarity_search(target_dir: &String) -> Result<()> {
    let target_dir = Path::new(target_dir);

    // Group photos by parent directory
    let mut photos_by_dir: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();
    // Walk through target directory
    for entry in WalkDir::new(target_dir).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            let path = entry.path().to_path_buf();
            let parent = path.parent().unwrap_or(Path::new("")).to_path_buf();
            let extension = path
                .extension()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase();
            if extension != "jpg" && extension != "jpeg" && extension != "png" {
                continue;
            }
            photos_by_dir.entry(parent).or_default().push(path.clone());
        }
    }

    let path = CString::new(format!(
        "{}/lib/libtorch_cuda.so",
        std::env::var("LIBTORCH").unwrap_or("/opt/libtorch".to_string())
    ))
    .unwrap();
    unsafe {
        dlopen(path.into_raw(), 1);
    }

    println!("Using CUDA: {}", tch::Cuda::is_available());
    println!("Using cuDNN: {}", tch::Cuda::cudnn_is_available());

    let device = tch::Device::cuda_if_available();
    let mut vs = tch::nn::VarStore::new(device);
    let net = Box::new(vit_small(&vs.root()));
    vs.load("assets/dinov2_vits14.safetensors")?;

    let mut always_yes = false;
    for (dir, photos) in &photos_by_dir {
        println!("Directory {:?}: {} photos", dir, photos.len());
        if photos.len() < 2 {
            continue;
        }
        println!("Calculating features for photos under directory {:?}", dir);
        let pb = ProgressBar::new(photos.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        let start = std::time::Instant::now();
        let features = photos
            .par_iter()
            .filter_map(|photo| {
                pb.inc(1);
                let img = tch::no_grad(|| {
                    imagenet::load_image_and_resize224(photo.to_str().unwrap())
                        .context(format!("Failed to load image {}", photo.display()))
                });
                if let Ok(img) = img { Some(img) } else { None }
            })
            .collect::<Vec<_>>()
            .chunks(64)
            .map(|chunk| {
                let tensor = tch::Tensor::stack(&chunk, 0).to(device);
                let output = tch::no_grad(|| net.extract_features(&tensor));
                output
            })
            .collect::<Vec<_>>();
        pb.finish();
        let duration = start.elapsed();
        println!("Time taken for feature extraction: {:?}", duration);

        let features_tensor = tch::Tensor::cat(&features, 0);
        println!("Features tensor shape: {:?}", features_tensor.size());
        let features_prod = features_tensor.mm(&features_tensor.transpose(1, 0));
        let features_norm = features_tensor.linalg_norm(2, -1, true, tch::Kind::Float);
        let features_norm_prod = features_norm.mm(&features_norm.transpose(1, 0));
        let similarity = features_prod / features_norm_prod;
        let n = similarity.size()[0] as usize;
        let m = similarity.size()[1] as usize;

        let mut similarity_data = vec![0f32; n * m];
        similarity.copy_data(&mut similarity_data, n * m);

        let mut similar_pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if similarity_data[i * m + j] > SIMILARITY_THRESHOLD {
                    similar_pairs.push((i, j, similarity_data[i * m + j]));
                }
            }
        }
        if similar_pairs.len() == 0 {
            println!("No similar images found");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            continue;
        }

        let mut files = Vec::new();
        for (i, j, score) in similar_pairs {
            let photo1 = &photos[i];
            let photo2 = &photos[j];

            // Get metadata for both photos, skip if either fails
            let (photo1_metadata, photo2_metadata) =
                match (fs::metadata(photo1), fs::metadata(photo2)) {
                    (Ok(m1), Ok(m2)) => (m1, m2),
                    _ => continue,
                };

            // Keep larger file, remove smaller one
            let (keep, remove, keep_len, remove_len) =
                if photo1_metadata.len() > photo2_metadata.len() {
                    (photo1, photo2, photo1_metadata.len(), photo2_metadata.len())
                } else {
                    (photo2, photo1, photo2_metadata.len(), photo1_metadata.len())
                };

            println!("\n📸 Similar images found (similarity: {:.3})", score);
            println!(
                "   Keeping:  {} ({:.3} MB)",
                style(keep.display()).green(),
                keep_len as f64 / 1024.0 / 1024.0
            );
            println!(
                "   Removing: {} ({:.3} MB)",
                style(remove.display()).red(),
                remove_len as f64 / 1024.0 / 1024.0
            );

            let name = match (keep.file_name(), remove.file_name()) {
                (Some(keep_name), Some(remove_name)) => {
                    if keep_name.len() < remove_name.len() {
                        keep_name.to_str()
                    } else {
                        remove_name.to_str()
                    }
                }
                _ => None,
            };
            files.push((keep, remove, name));
        }

        if !always_yes {
            println!("\n❓ Would you like to proceed with removing these similar files? [y/n/a]");
            println!("   y: yes for this group");
            println!("   n: no for skip this group");
            println!("   a: yes for all remaining groups");

            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
            let input = input.trim().to_lowercase();

            match input.as_str() {
                "y" => (), // Continue with current group
                "a" => always_yes = true,
                _ => {
                    continue;
                }
            }
        }

        for (_, to_remove, _) in files.iter() {
            match trash::delete(to_remove) {
                Ok(_) => println!("🗑️ Moved to trash: {:?}", to_remove),
                _ => continue,
            }
        }

        for (to_keep, _, new_name) in files.iter() {
            if let Some(new_name) = new_name {
                let dest_path = to_keep.parent().unwrap().join(new_name);
                if dest_path == **to_keep {
                    continue;
                }
                match fs::rename(to_keep, &dest_path) {
                    Ok(_) => println!(
                        "✅ Renamed {} → {}",
                        style(to_keep.display()).dim(),
                        style(dest_path.display()).green()
                    ),
                    _ => continue,
                }
            }
        }

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    Ok(())
}

fn remove_duplicates(photos: &mut HashMap<String, PhotoInfo>) {
    // Group photos by hash
    let mut hash_groups: HashMap<String, Vec<String>> = HashMap::new();
    for (path, photo) in photos.iter() {
        hash_groups
            .entry(photo.hash.clone())
            .or_default()
            .push(path.clone());
    }

    // Count duplicate groups and total duplicate images
    let dup_groups = hash_groups.values().filter(|paths| paths.len() > 1).count();
    let dup_images: usize = hash_groups
        .values()
        .filter(|paths| paths.len() > 1)
        .map(|paths| paths.len())
        .sum();

    println!(
        "\n🔍 Found {} duplicate groups containing {} duplicate images in total",
        style(dup_groups).bold(),
        style(dup_images).bold()
    );

    // For each hash group, keep photo with shortest filename and remove others
    for paths in hash_groups.values() {
        if paths.len() <= 1 {
            continue;
        }

        // Find path with shortest filename
        let shortest = paths
            .iter()
            .min_by_key(|path| {
                Path::new(path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.len())
                    .unwrap_or(usize::MAX)
            })
            .unwrap();

        // Remove all paths except shortest
        for path in paths {
            if path != shortest {
                photos.remove(path);
            }
        }
    }
}

fn print_stats(photos: &HashMap<String, PhotoInfo>) -> Result<()> {
    // Calculate total size
    let total_size: u64 = photos
        .iter()
        .map(|(_, photo)| fs::metadata(&photo.path).map(|m| m.len()).unwrap_or(0))
        .sum();

    // Count photos per year
    let mut year_counts: HashMap<i32, usize> = HashMap::new();
    for photo in photos.values() {
        let year = photo.date.year();
        *year_counts.entry(year).or_insert(0) += 1;
    }

    // Count extensions
    let mut ext_counts: HashMap<String, usize> = HashMap::new();
    for photo in photos.values() {
        if let Some(ext) = photo.path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            *ext_counts.entry(ext.to_string()).or_insert(0) += 1;
        }
    }

    // Print statistics
    println!("📊 Total photos: {}", style(photos.len()).bold());
    println!(
        "💾 Total size: {:.2} GB",
        style(total_size as f64 / 1_073_741_824.0).bold()
    );

    println!("\n📅 Photos by Year:");
    println!("┌──────┬───────────┐");
    println!("│ Year │   Photos  │");
    println!("├──────┼───────────┤");
    let mut years: Vec<_> = year_counts.iter().collect();
    years.sort_by_key(|&(k, _)| k);
    for (year, count) in years {
        println!("│ {:4} │ {:>9} │", style(year).bold(), style(count).bold());
    }
    println!("└──────┴───────────┘");

    println!("\n🗂️  Photos by Extension:");
    println!("┌──────────┬───────────┐");
    println!("│   Ext    │   Files   │");
    println!("├──────────┼───────────┤");
    let mut exts: Vec<_> = ext_counts.iter().collect();
    exts.sort_by_key(|&(_, count)| std::cmp::Reverse(*count));
    for (ext, count) in exts {
        println!("│ .{:<7} │ {:>9} │", style(ext).bold(), style(count).bold());
    }
    println!("└──────────┴───────────┘");

    Ok(())
}

fn collect_photos(source_dir: &Path, media_type: &Media) -> impl Iterator<Item = PathBuf> {
    WalkDir::new(source_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter_map(|entry| {
            let path = entry.path();
            let extension = path.extension()?;
            let ext = extension.to_string_lossy().to_lowercase();

            match get_media_type(&ext) {
                t if t == *media_type => Some(path.to_path_buf()),
                _ => None,
            }
        })
}

fn calculate_hash(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("Failed to open file for hashing: {}", path.display()))?;

    let mut hasher = blake3::Hasher::new();
    std::io::copy(&mut file, &mut hasher)
        .with_context(|| format!("Failed to read file for hashing: {}", path.display()))?;

    Ok(hasher.finalize().to_hex().to_string())
}

fn get_exif_info(path: &Path) -> Option<HashMap<String, String>> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open file {}: {}", path.display(), e);
            return None;
        }
    };
    let mut bufreader = BufReader::new(&file);

    match Reader::new().read_from_container(&mut bufreader) {
        Ok(exif) => {
            let exif_map = exif
                .fields()
                .filter_map(|field| {
                    // Skip MakerNote and Tiff tags since they're usually not useful
                    match field.tag.to_string() {
                        tag if tag.contains("MakerNote") || tag.starts_with("Tag(Tiff,") => None,
                        tag => Some((tag, field.display_value().to_string())),
                    }
                })
                .collect();

            Some(exif_map)
        }
        Err(_) => {
            // eprintln!("Error reading EXIF metadata: {} for {}", e, path.display());
            None
        }
    }
}

fn get_date_from_exif(exif: &Option<HashMap<String, String>>) -> Option<NaiveDateTime> {
    // 如果没有EXIF信息，返回Unix纪元时间
    exif.as_ref()
        .and_then(|exif| exif.get("DateTimeOriginal"))
        .or_else(|| exif.as_ref().and_then(|e| e.get("DateTime")))
        .or_else(|| exif.as_ref().and_then(|e| e.get("CreateDate")))
        .and_then(|datetime_str| {
            NaiveDateTime::parse_from_str(datetime_str, "%Y-%m-%d %H:%M:%S").ok()
        })
        .or(Some(NaiveDateTime::UNIX_EPOCH))
}

fn get_media_type(extension: &str) -> Media {
    let ext = extension.to_lowercase();
    let photo_exts = [
        // Common web/general formats
        "jpg", "jpeg", "png", "gif", "bmp", "webp", "ico", "pcx",
        // Professional RAW formats
        "raw", "arw", "cr2", "cr3", "nef", "dng", "orf", "rw2", "pef", "srw", "raf", "x3f", "crw",
        "erf", "kdc", "dcr", "mrw", "nrw", "ptx", "rwl", "srf", // Modern formats
        "heic", "heif", "svg", "avif", "jxl", "jp2", "j2k", "jpf",
        // Professional editing formats
        "tiff", "tif", "psd", "psb", "xcf", "ai", "eps", "indd", // HDR formats
        "hdr", "exr", "dpx",
    ];

    let video_exts = [
        // Common web/consumer formats
        "mp4", "mov", "avi", "wmv", "flv", "f4v", "swf", // Professional/Cinema formats
        "mxf", "r3d", "braw", "ari", "dcp", "dpx", "prores", // Modern container formats
        "mkv", "webm", "m4v", "mpg", "mpeg", "m2v", "m2p", // Mobile/legacy formats
        "3gp", "3g2", "mts", "m2ts", "ts", "asf", "dv", "vid", // Other formats
        "vob", "ogv", "rm", "rmvb", "divx", "xvid", "y4m", "yuv",
        // Professional editing formats
        "aaf", "dnxhd", "cine", "ifo", "vro", "evo",
    ];

    if photo_exts.contains(&ext.as_str()) {
        Media::Photo
    } else if video_exts.contains(&ext.as_str()) {
        Media::Video
    } else {
        Media::Other
    }
}
