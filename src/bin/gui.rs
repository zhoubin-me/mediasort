use anyhow::Context;
use chrono::{Datelike, NaiveDateTime};
use console::style;
use glib::{MainContext, clone};
use gtk::{Application, gio, prelude::*};
use libc::dlopen;
use mediasort::utils::*;
use rayon::prelude::*;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::CString;
use std::rc::Rc;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
};
use tch::vision::dinov2::vit_small;
use tch::vision::imagenet;
use walkdir::WalkDir;

//-------------------------------------------------------------------------------
// Main Application Entry Point
//-------------------------------------------------------------------------------

fn main() {
    let application = Application::new(
        Some("com.example.mediasort"),
        gio::ApplicationFlags::FLAGS_NONE,
    );

    application.connect_activate(|app| {
        App::create(app, None);
    });

    application.run();
}

//-------------------------------------------------------------------------------
// Data Structures
//-------------------------------------------------------------------------------

#[derive(Debug, Default)]
struct AppState {
    source_dir: Option<PathBuf>,
    target_dir: Option<PathBuf>,
    scan_photos: bool,
    photos: HashMap<String, PhotoInfo>,
    continue_review: bool,
}

#[derive(Debug, Clone)]
enum Event {
    SourceDirectorySelected(PathBuf),
    TargetDirectorySelected(PathBuf),
    MediaTypeChanged(bool),
    StartScan,
    ScanProgress(PhotoInfo, usize, usize),
    ScanComplete(usize),
    RemoveDuplicates,
    DuplicatesRemoved(usize, usize),
    SortAndCopy,
    SortProgress(usize, usize),
    SortComplete(usize),
    FindSimilar,
    SimilarFound(Vec<(PathBuf, PathBuf, f32)>),
    SimilarProgress(usize, usize, String),
    SimilarProcessed,
    ReviewSimilarPair(usize),
    KeepImage(usize, PathBuf, PathBuf),
    SkipPair(usize),
    AutoProcessRemaining,
    SimilarReviewComplete,
}

//-------------------------------------------------------------------------------
// Widgets and Controllers
//-------------------------------------------------------------------------------

#[derive(Clone)]
struct Widgets {
    window: gtk::ApplicationWindow,
    source_path_entry: gtk::Entry,
    source_button: gtk::Button,
    target_path_entry: gtk::Entry,
    target_button: gtk::Button,
    media_switch: gtk::Switch,
    media_type_label: gtk::Label,
    scan_button: gtk::Button,
    remove_duplicates_button: gtk::Button,
    sort_copy_button: gtk::Button,
    find_similar_button: gtk::Button,
    progress_bar: gtk::ProgressBar,
    status_label: gtk::Label,
    review_box: gtk::Box,
    keep_first_button: Option<gtk::Button>,
    keep_second_button: Option<gtk::Button>,
    skip_button: Option<gtk::Button>,
    auto_button: Option<gtk::Button>,
    score_label: Option<gtk::Label>,
}

impl Widgets {
    fn init(application: &Application) -> Self {
        // Create main window and container
        let window = gtk::ApplicationWindow::new(application);
        window.set_title(Some("Media Sort"));
        window.set_default_size(900, 1200);

        let main_box = gtk::Box::new(gtk::Orientation::Vertical, 10);
        main_box.set_margin_start(10);
        main_box.set_margin_end(10);
        main_box.set_margin_top(10);
        main_box.set_margin_bottom(10);
        window.set_child(Some(&main_box));

        // Build UI Components
        let (source_path_entry, source_button) =
            create_directory_selector("Source Directory:", &main_box);
        let (target_path_entry, target_button) =
            create_directory_selector("Target Directory:", &main_box);
        let (media_switch, media_type_label) = create_media_type_selector(&main_box);

        let button_section = create_action_buttons(&main_box);
        let scan_button = button_section.0;
        let remove_duplicates_button = button_section.1;
        let sort_copy_button = button_section.2;
        let find_similar_button = button_section.3;

        // Create review box (initially hidden)
        let review_box = gtk::Box::new(gtk::Orientation::Vertical, 10);
        review_box.set_margin_top(10);
        review_box.set_margin_bottom(10);
        review_box.set_visible(true); // Initially hidden
        // Add a spacer to push status elements to the bottom
        let spacer = gtk::Box::new(gtk::Orientation::Vertical, 0);
        spacer.set_vexpand(true);
        review_box.append(&spacer);

        main_box.append(&review_box);

        // Create status area
        let status_area = gtk::Box::new(gtk::Orientation::Vertical, 5);
        status_area.set_margin_top(10);

        let progress_bar = gtk::ProgressBar::new();
        progress_bar.set_show_text(true);
        status_area.append(&progress_bar);

        let status_label = gtk::Label::new(Some("Ready"));
        status_label.set_halign(gtk::Align::Start);
        status_area.append(&status_label);

        main_box.append(&status_area);

        Widgets {
            window,
            source_path_entry,
            source_button,
            target_path_entry,
            target_button,
            media_switch,
            media_type_label,
            scan_button,
            remove_duplicates_button,
            sort_copy_button,
            find_similar_button,
            progress_bar,
            status_label,
            review_box,
            keep_first_button: None,
            keep_second_button: None,
            skip_button: None,
            auto_button: None,
            score_label: None,
        }
    }

    fn disable_widgets(&self) {
        self.source_path_entry.set_sensitive(false);
        self.source_button.set_sensitive(false);
        self.target_path_entry.set_sensitive(false);
        self.target_button.set_sensitive(false);
        self.scan_button.set_sensitive(false);
        self.remove_duplicates_button.set_sensitive(false);
        self.sort_copy_button.set_sensitive(false);
        self.find_similar_button.set_sensitive(false);
        self.source_path_entry.set_sensitive(false);
        self.target_path_entry.set_sensitive(false);
        self.media_switch.set_sensitive(false);
    }

    fn enable_widgets(&self) {
        self.source_path_entry.set_sensitive(true);
        self.source_button.set_sensitive(true);
        self.target_path_entry.set_sensitive(true);
        self.target_button.set_sensitive(true);
        self.scan_button.set_sensitive(true);
        self.remove_duplicates_button.set_sensitive(true);
        self.sort_copy_button.set_sensitive(true);
        self.find_similar_button.set_sensitive(true);
        self.source_path_entry.set_sensitive(true);
        self.target_path_entry.set_sensitive(true);
        self.media_switch.set_sensitive(true);
    }
}

//-------------------------------------------------------------------------------
// Main Application Structure
//-------------------------------------------------------------------------------

struct App {
    widgets: Widgets,
    state: Arc<Mutex<AppState>>,
    sender: glib::Sender<Event>,
    similar_pairs: Rc<RefCell<Vec<(PathBuf, PathBuf, f32)>>>,
    current_pair_index: Rc<Cell<usize>>,
}

impl App {
    pub fn create(application: &gtk::Application, _file: Option<&gio::File>) {
        let widgets = Widgets::init(application);

        let state = Arc::new(Mutex::new(AppState {
            scan_photos: true,
            ..AppState::default()
        }));

        let (sender, receiver) = MainContext::channel::<Event>(glib::Priority::DEFAULT);

        let mut app = Self {
            widgets,
            state,
            sender,
            similar_pairs: Rc::new(RefCell::new(Vec::new())),
            current_pair_index: Rc::new(Cell::new(0)),
        };

        // Connect events
        app.connect_events();

        // Show the window
        app.widgets.window.present();

        // Set up event processing
        receiver.attach(None, move |event| {
            app.process_event(event);
            glib::ControlFlow::Continue
        });
    }

    fn connect_events(&self) {
        let sender = self.sender.clone();

        // Connect source directory button
        self.widgets
            .source_path_entry
            .connect_changed(clone!(@strong sender => move |entry| {
                let path_text = entry.text().to_string();
                if !path_text.is_empty() {
                    let path = PathBuf::from(&path_text);
                    if path.exists() && path.is_dir() {
                        let _ = sender.send(Event::SourceDirectorySelected(path));
                    }
                }
            }));

        // Connect target directory button
        self.widgets
            .target_path_entry
            .connect_changed(clone!(@strong sender => move |entry| {
                let path_text = entry.text().to_string();
                if !path_text.is_empty() {
                    let path = PathBuf::from(&path_text);
                    if path.exists() && path.is_dir() {
                        let _ = sender.send(Event::TargetDirectorySelected(path));
                    }
                }
            }));

        // Connect media switch
        self.widgets
            .media_switch
            .connect_state_set(clone!(@strong sender => move |_, is_active| {
                let _ = sender.send(Event::MediaTypeChanged(is_active));
                glib::Propagation::Stop
            }));

        // Connect scan button
        self.widgets
            .scan_button
            .connect_clicked(clone!(@strong sender => move |_| {
                let _ = sender.send(Event::StartScan);
            }));

        // Connect remove duplicates button
        self.widgets
            .remove_duplicates_button
            .connect_clicked(clone!(@strong sender => move |_| {
                let _ = sender.send(Event::RemoveDuplicates);
            }));

        // Connect sort and copy button
        self.widgets
            .sort_copy_button
            .connect_clicked(clone!(@strong sender => move |_| {
                let _ = sender.send(Event::SortAndCopy);
            }));

        // Connect find similar button
        self.widgets
            .find_similar_button
            .connect_clicked(clone!(@strong sender => move |_| {
                let _ = sender.send(Event::FindSimilar);
            }));

        // Connect directory browse buttons
        self.connect_directory_button(
            &self.widgets.source_button,
            &self.widgets.source_path_entry,
            "Select Source Directory",
            Event::SourceDirectorySelected,
        );

        self.connect_directory_button(
            &self.widgets.target_button,
            &self.widgets.target_path_entry,
            "Select Target Directory",
            Event::TargetDirectorySelected,
        );
    }

    fn connect_directory_button<F>(
        &self,
        button: &gtk::Button,
        path_entry: &gtk::Entry,
        dialog_title: &str,
        event_constructor: F,
    ) where
        F: Fn(PathBuf) -> Event + 'static + Clone,
    {
        let window = self.widgets.window.clone();
        let path_entry = path_entry.clone();
        let sender = self.sender.clone();
        let dialog_title = dialog_title.to_string();

        button.connect_clicked(
            clone!(@strong window, @strong path_entry, @strong sender, @strong event_constructor, @strong dialog_title => move |_| {
                let file_chooser = gtk::FileDialog::new();
                file_chooser.set_title(&dialog_title);
                file_chooser.set_modal(true);

                file_chooser.select_folder(
                    Some(&window),
                    None::<&gio::Cancellable>,
                    clone!(@strong path_entry, @strong sender, @strong event_constructor => move |result| {
                        if let Ok(file) = result {
                            if let Some(path) = file.path() {
                                path_entry.set_text(&path.display().to_string());
                                let _ = sender.send(event_constructor(path));
                            }
                        }
                    }),
                );
            }),
        );
    }

    fn process_event(&mut self, event: Event) {
        match event {
            Event::SourceDirectorySelected(path) => {
                let mut state = self.state.lock().unwrap();
                state.source_dir = Some(path.clone());
                state.photos.clear();
                self.widgets
                    .status_label
                    .set_text(&format!("Source directory: {}", path.display()));
            }
            Event::TargetDirectorySelected(path) => {
                let mut state = self.state.lock().unwrap();
                state.target_dir = Some(path.clone());
                self.widgets
                    .status_label
                    .set_text(&format!("Target directory: {}", path.display()));
            }
            Event::MediaTypeChanged(is_photos) => {
                let mut state = self.state.lock().unwrap();
                state.scan_photos = is_photos;
                if is_photos {
                    self.widgets.media_type_label.set_text("Photos");
                } else {
                    self.widgets.media_type_label.set_text("Videos");
                }
            }
            Event::StartScan => {
                self.widgets.disable_widgets();
                self.start_scan();
            }
            Event::ScanProgress(photo_info, current, total) => {
                let mut state = self.state.lock().unwrap();
                let progress = current as f64 / total as f64;

                self.widgets.progress_bar.set_fraction(progress);
                self.widgets
                    .status_label
                    .set_text(&format!("Processed {}/{}", current, total));

                state
                    .photos
                    .insert(photo_info.path.display().to_string(), photo_info);
            }
            Event::ScanComplete(total) => {
                self.widgets.enable_widgets();
                self.widgets.progress_bar.set_fraction(1.0);
                self.widgets
                    .status_label
                    .set_text(&format!("Scan complete! Processed {} files.", total));
            }
            Event::RemoveDuplicates => {
                self.widgets.disable_widgets();
                self.remove_duplicates();
            }
            Event::DuplicatesRemoved(dup_groups, dup_images) => {
                self.widgets.enable_widgets();
                self.widgets.status_label.set_text(&format!(
                    "Found {} duplicate groups and removed {} duplicate images in total",
                    dup_groups, dup_images
                ));
            }
            Event::SortAndCopy => {
                self.widgets.disable_widgets();
                self.sort_and_copy();
            }
            Event::SortProgress(processed, total) => {
                let progress = processed as f64 / total as f64;
                self.widgets.progress_bar.set_fraction(progress);
                self.widgets.status_label.set_text(&format!(
                    "Copied {}/{} files to target directory",
                    processed, total
                ));
            }
            Event::SortComplete(total) => {
                self.widgets.enable_widgets();
                self.widgets.progress_bar.set_fraction(1.0);
                self.widgets.status_label.set_text(&format!(
                    "Sort and copy complete! Processed {} files.",
                    total
                ));
            }
            Event::FindSimilar => {
                self.widgets.disable_widgets();
                self.find_similar();
            }
            Event::SimilarFound(similar_pairs) => {
                if similar_pairs.is_empty() {
                    self.widgets
                        .status_label
                        .set_text("No similar images found.");
                    self.widgets.progress_bar.set_fraction(0.0);
                    let _ = self.sender.send(Event::SimilarProcessed);
                    return;
                }

                // Store the pairs and reset the index
                *self.similar_pairs.borrow_mut() = similar_pairs;
                self.current_pair_index.set(0);

                // Show the first pair
                let similar_pairs_data = self.similar_pairs.borrow().clone();
                if !similar_pairs_data.is_empty() {
                    let _ = self.sender.send(Event::ReviewSimilarPair(0));
                }

                self.widgets.status_label.set_text(&format!(
                    "Found {} similar image pairs. Reviewing...",
                    similar_pairs_data.len()
                ));
            }
            Event::ReviewSimilarPair(index) => {
                println!("Reviewing pair index {}", index);

                self.create_similar_review_ui(index);
                let index = self.current_pair_index.get();
                let pairs_len = self.similar_pairs.borrow().len();
                if index >= pairs_len {
                    let _ = self.sender.send(Event::SimilarProcessed);
                }
            }
            Event::KeepImage(index, path_to_keep, path_to_remove) => {
                // Delete the image that's not kept
                let pairs = self.similar_pairs.borrow();
                if index < pairs.len() {
                    // Delete the file
                    let name = match (path_to_keep.file_name(), path_to_remove.file_name()) {
                        (Some(keep_name), Some(remove_name)) => {
                            if keep_name.len() < remove_name.len() {
                                keep_name.to_str()
                            } else {
                                remove_name.to_str()
                            }
                        }
                        _ => None,
                    };

                    match trash::delete(path_to_remove.clone()) {
                        Ok(_) => println!("🗑️ Moved to trash: {:?}", path_to_remove),
                        _ => (),
                    }

                    if let Some(new_name) = name {
                        let dest_path = path_to_keep.parent().unwrap().join(new_name);
                        if dest_path != path_to_keep {
                            match std::fs::rename(path_to_keep.clone(), &dest_path) {
                                Ok(_) => println!(
                                    "✅ Renamed {} → {}",
                                    style(path_to_keep.clone().display()).dim(),
                                    style(dest_path.display()).green()
                                ),
                                _ => (),
                            }
                        }
                    }

                    if let Err(e) = std::fs::remove_file(path_to_remove.clone()) {
                        eprintln!("Failed to delete {}: {}", path_to_remove.display(), e);
                    } else {
                        println!("Deleted {}", path_to_remove.display());
                    }
                }
                // Instead of calling show_next_pair directly, send an event
                let _ = self.sender.send(Event::SkipPair(index));
            }
            Event::SkipPair(index) => {
                // Update the current index
                self.current_pair_index.set(index + 1);
                let next_index = index + 1;
                // Check if we have more pairs to review
                let pairs_len = self.similar_pairs.borrow().len();
                if next_index < pairs_len {
                    // Get the next pair
                    {
                        let pairs = self.similar_pairs.borrow();
                        (
                            pairs[next_index].0.clone(),
                            pairs[next_index].1.clone(),
                            pairs[next_index].2,
                        )
                    };

                    // Show the next pair
                    let _ = self.sender.send(Event::ReviewSimilarPair(next_index));
                } else {
                    let _ = self.sender.send(Event::SimilarProcessed);
                }
            }
            Event::AutoProcessRemaining => {
                // Auto process all remaining pairs (keep the first image of each pair)
                let pairs = self.similar_pairs.borrow();
                let current = self.current_pair_index.get();

                for i in current..pairs.len() {
                    let (photo1, photo2, _score) = &pairs[i];

                    // Delete the second image
                    let (photo1_metadata, photo2_metadata) =
                        match (std::fs::metadata(photo1), std::fs::metadata(photo2)) {
                            (Ok(m1), Ok(m2)) => (m1, m2),
                            _ => continue,
                        };

                    // Keep larger file, remove smaller one
                    let (keep, remove, _, _) = if photo1_metadata.len() > photo2_metadata.len() {
                        (photo1, photo2, photo1_metadata.len(), photo2_metadata.len())
                    } else {
                        (photo2, photo1, photo2_metadata.len(), photo1_metadata.len())
                    };

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

                    match trash::delete(remove) {
                        Ok(_) => println!("🗑️ Moved to trash: {:?}", remove),
                        _ => continue,
                    }

                    if let Some(new_name) = name {
                        let dest_path = keep.parent().unwrap().join(new_name);
                        if dest_path == **keep {
                            continue;
                        }
                        match std::fs::rename(keep, &dest_path) {
                            Ok(_) => println!(
                                "✅ Renamed {} → {}",
                                style(keep.display()).dim(),
                                style(dest_path.display()).green()
                            ),
                            _ => continue,
                        }
                    }
                }
                let _ = self.sender.send(Event::SimilarProcessed);
            }
            Event::SimilarProcessed => {
                while let Some(child) = self.widgets.review_box.last_child() {
                    self.widgets.review_box.remove(&child);
                }
                let spacer = gtk::Box::new(gtk::Orientation::Vertical, 0);
                spacer.set_vexpand(true);
                self.widgets.review_box.append(&spacer);
                self.state.lock().unwrap().continue_review = true;
            }
            Event::SimilarProgress(current, total, status) => {
                let progress_fraction = current as f64 / total as f64;
                self.widgets.progress_bar.set_fraction(progress_fraction);
                self.widgets.status_label.set_text(&status);
            }

            Event::SimilarReviewComplete => {
                while let Some(child) = self.widgets.review_box.last_child() {
                    self.widgets.review_box.remove(&child);
                }
                let spacer = gtk::Box::new(gtk::Orientation::Vertical, 0);
                spacer.set_vexpand(true);
                self.widgets.review_box.append(&spacer);

                self.widgets.enable_widgets();
                self.widgets
                    .status_label
                    .set_text("Similar image processing complete!");
            }
        }
    }

    fn start_scan(&self) {
        let mut state = self.state.lock().unwrap();

        if state.source_dir.is_none() {
            self.widgets
                .status_label
                .set_text("Please select source directory first.");
            self.widgets.enable_widgets();
            return;
        }

        let source_dir = state.source_dir.clone().unwrap();
        let scan_photos = state.scan_photos;

        // Reset UI
        self.widgets.progress_bar.set_fraction(0.0);

        // Clear previous photos
        state.photos.clear();

        let (media_type, cache_path) = if scan_photos {
            (&Media::Photo, source_dir.join(".photo_cache"))
        } else {
            (&Media::Video, source_dir.join(".video_cache"))
        };

        // Count total files
        let total_files = collect_photos(&source_dir, &media_type).count();

        // Create sender for worker thread
        let sender = self.sender.clone();

        // Start worker thread
        let source_dir_clone = source_dir.clone();
        let cache_path_clone = cache_path.clone();
        let processed = Arc::new(Mutex::new(0usize));

        thread::spawn(move || {
            let cached_photos = load_cached_photos(&cache_path_clone);

            collect_photos(&source_dir_clone, &media_type)
                .par_bridge()
                .filter_map(|path| {
                    let path_str = path.display().to_string();
                    let n: usize = {
                        let mut processed = processed.lock().unwrap();
                        *processed += 1;
                        *processed
                    };
                    if let Some(cached_info) = cached_photos.get(&path_str) {
                        // Check if file size and modified time match to validate cache
                        if let Ok(metadata) = std::fs::metadata(&path) {
                            if metadata.len() == cached_info.size {
                                let _ = sender.send(Event::ScanProgress(
                                    cached_info.clone(),
                                    n,
                                    total_files,
                                ));
                                return Some(());
                            }
                        }
                    }

                    let hash = match calculate_hash(&path, media_type) {
                        Ok(hash) => hash,
                        _ => format!("random_{:x}", rand::random::<u128>()),
                    };

                    let exif = match get_exif_info(&path, media_type) {
                        Ok(Some(exif)) => Some(exif),
                        _ => None,
                    };

                    let date = get_date_from_exif(&exif).unwrap();
                    let size = std::fs::metadata(&path).unwrap().len();

                    let photo_info = PhotoInfo {
                        path,
                        hash,
                        date,
                        size,
                    };

                    let _ = sender.send(Event::ScanProgress(photo_info, n, total_files));
                    Some(())
                })
                .collect::<Vec<_>>();

            // Send completion message
            let _ = sender.send(Event::ScanComplete(total_files));
        });
    }

    fn remove_duplicates(&self) {
        let state_guard = self.state.lock().unwrap();

        if state_guard.photos.is_empty() {
            self.widgets
                .status_label
                .set_text("No photos to process. Please scan first.");
            self.widgets.enable_widgets();

            return;
        }

        self.widgets.status_label.set_text("Finding duplicates...");
        self.widgets.remove_duplicates_button.set_sensitive(false);

        // Determine cache file based on scan_photos flag
        let cache_file = if state_guard.scan_photos {
            ".photo_cache"
        } else {
            ".video_cache"
        };
        let cache_path = state_guard.source_dir.as_ref().unwrap().join(cache_file);

        let mut photos = state_guard.photos.clone();
        let sender = self.sender.clone();
        let state_arc = self.state.clone();

        thread::spawn(move || {
            save_cached_photos(&cache_path, &photos).expect("Failed to save cached photos");
            let results = remove_duplicates(&mut photos);

            // Update state with new photos
            let mut state = state_arc.lock().unwrap();
            state.photos = photos;

            // Send results back
            let _ = sender.send(Event::DuplicatesRemoved(results.0, results.1));
        });
    }

    fn sort_and_copy(&self) {
        let state = self.state.lock().unwrap();

        if state.photos.is_empty() {
            self.widgets
                .status_label
                .set_text("No media files to process. Please scan first.");
            self.widgets.enable_widgets();
            return;
        }

        if state.target_dir.is_none() {
            self.widgets
                .status_label
                .set_text("Please select a target directory first.");
            self.widgets.enable_widgets();
            return;
        }

        let photos = state.photos.clone();
        let target_dir = state.target_dir.clone().unwrap();

        self.widgets
            .status_label
            .set_text("Sorting and copying files to target directory...");
        self.widgets.progress_bar.set_fraction(0.0);
        self.widgets.sort_copy_button.set_sensitive(false);

        let sender = self.sender.clone();
        let total = photos.len();
        let processed = Arc::new(Mutex::new(0usize));

        thread::spawn(move || {
            let result = photos.values().par_bridge().try_for_each(|photo| {
                // Create YYYY-MM directory structure
                let year = photo.date.year();
                let month = photo.date.month();
                let target_subdir = target_dir.join(format!("{:04}-{:02}", year, month));

                if let Ok(metadata) = std::fs::metadata(&photo.path) {
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
                std::fs::create_dir_all(&target_subdir).with_context(|| {
                    format!("Failed to create directory: {}", target_subdir.display())
                })?;

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

                std::fs::copy(&photo.path, &target_path).with_context(|| {
                    format!(
                        "Failed to copy {} to {}",
                        photo.path.display(),
                        target_path.display()
                    )
                })?;

                let x = {
                    let mut processed = processed.lock().unwrap();
                    *processed += 1;
                    *processed
                };
                let _ = sender.send(Event::SortProgress(x, total));
                Ok::<(), anyhow::Error>(())
            });

            if let Err(e) = result {
                eprintln!("Error sorting and copying files: {}", e);
            }

            let _ = sender.send(Event::SortComplete(total));
        });
    }

    fn find_similar(&self) {
        let target_dir = {
            let state = self.state.lock().unwrap();
            if state.target_dir.is_none() {
                self.widgets.enable_widgets();
                self.widgets
                    .status_label
                    .set_text("Please select a target directory first.");
                return;
            }
            state.target_dir.clone().unwrap()
        };

        let sender = self.sender.clone();
        self.widgets
            .status_label
            .set_text("Finding similar images...");
        self.widgets.progress_bar.set_fraction(0.0);

        let state_clone = self.state.clone();
        let sender_clone = sender.clone();
        thread::spawn(move || {
            // Find all image files in target directory recursively
            let mut photos_by_dir: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();
            // Walk through each subdirectory
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
            vs.load("assets/dinov2_vits14.safetensors")
                .expect("Failed to load model");

            for (index, (dir, photos)) in photos_by_dir.iter().enumerate() {
                if photos.len() < 2 {
                    continue;
                }

                let total = photos.len();
                let processed = Arc::new(Mutex::new(0usize));
                let data= photos
                    .par_iter()
                    .filter_map(|photo| {
                        let x = {
                            let mut processed = processed.lock().unwrap();
                            *processed += 1;
                            *processed
                        };
                        let _ = sender_clone.send(Event::SimilarProgress(
                            x,
                            total,
                            format!("Calculating features for photos under directory {:?}: {}/{} processed directories", dir, index + 1, photos_by_dir.len()),
                        ));
                        let img = tch::no_grad(|| {
                            imagenet::load_image_and_resize224(photo.to_str().unwrap())
                                .context(format!("Failed to load image {}", photo.display()))
                        });
                        if let Ok(img) = img { Some((img, photo)) } else { None }
                    })
                    .collect::<Vec<_>>()
                    .chunks(64)
                    .map(|chunks| {
                        let tensors = chunks.iter().map(|(img, _)| img).collect::<Vec<_>>();
                        let photos = chunks.iter().map(|(_, photo)| *photo).collect::<Vec<_>>();

                        let tensor = tch::Tensor::stack(&tensors, 0).to(device);
                        let output = tch::no_grad(|| net.extract_features(&tensor));
                        (photos, output)
                    })
                    .collect::<Vec<_>>();

                let features = data.iter().map(|(_, output)| output).collect::<Vec<_>>();
                let photos = data.iter().map(|(photos, _)| photos).flatten().copied().collect::<Vec<_>>();

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
                            similar_pairs.push((
                                photos[i].clone(),
                                photos[j].clone(),
                                similarity_data[i * m + j],
                            ));
                        }
                    }
                }
                sender_clone
                    .send(Event::SimilarFound(similar_pairs))
                    .expect("Failed to send similar pairs");
                {
                    let mut state = state_clone.lock().unwrap();
                    state.continue_review = false;
                }

                loop {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    {
                        let state = state_clone.lock().unwrap();
                        if state.continue_review {
                            break;
                        }
                    }
                }
            }
            println!("Similar review complete");
            sender_clone
                .send(Event::SimilarReviewComplete)
                .expect("Failed to send similar pairs");
        });
    }

    fn create_similar_review_ui(&mut self, pair_index: usize) {
        // Clear the review box
        while let Some(child) = self.widgets.review_box.last_child() {
            self.widgets.review_box.remove(&child);
        }

        // Create a scrolled window to contain all pairs
        let scrolled_window = gtk::ScrolledWindow::new();
        scrolled_window.set_vexpand(true);
        scrolled_window.set_hexpand(true);
        scrolled_window.set_min_content_height(500);

        // Create a vertical box to hold all the pairs
        let pairs_box = gtk::Box::new(gtk::Orientation::Vertical, 10);
        pairs_box.set_margin_start(10);
        pairs_box.set_margin_end(10);
        pairs_box.set_margin_top(10);
        pairs_box.set_margin_bottom(10);

        scrolled_window.set_child(Some(&pairs_box));
        self.widgets.review_box.append(&scrolled_window);

        // Add all similar pairs to the scrollable box
        let similar_pairs_data = self.similar_pairs.borrow().clone();
        for (i, (path1, path2, sim_score)) in similar_pairs_data.clone().into_iter().enumerate() {
            // Create a frame for each pair
            let pair_frame = gtk::Frame::new(None);
            pair_frame.set_margin_bottom(15);

            // Highlight the current pair
            if i == pair_index {
                pair_frame.add_css_class("selected-pair");
                pair_frame.set_label(Some("Current Pair"));
            }

            let pair_box = gtk::Box::new(gtk::Orientation::Vertical, 5);
            pair_box.set_margin_start(10);
            pair_box.set_margin_end(10);
            pair_box.set_margin_top(10);
            pair_box.set_margin_bottom(10);

            // Add similarity score at the top of each pair
            let score_label = gtk::Label::new(None);
            score_label.set_markup(&format!(
                "<b>Pair #{}: Similarity Score: {:.2}%</b>",
                i + 1,
                sim_score * 100.0
            ));
            score_label.set_margin_bottom(5);
            pair_box.append(&score_label);

            // Create image container
            let image_box = gtk::Box::new(gtk::Orientation::Horizontal, 10);
            image_box.set_homogeneous(true);

            // Determine which image is larger to place on the left
            let (left_path, right_path) = match (
                std::fs::metadata(path1.clone()),
                std::fs::metadata(path2.clone()),
            ) {
                (Ok(meta1), Ok(meta2)) => {
                    if meta1.len() >= meta2.len() {
                        (path1.clone(), path2.clone())
                    } else {
                        (path2.clone(), path1.clone())
                    }
                }
                _ => (path1, path2), // Default if metadata can't be read
            };

            // Left image with label
            let left_box = gtk::Box::new(gtk::Orientation::Vertical, 5);
            let left_size = std::fs::metadata(left_path.clone())
                .map(|m| m.len() as f64 / (1024.0 * 1024.0))
                .unwrap_or(0.0);

            let left_label = gtk::Label::new(Some(&format!(
                "Image 1: {} (Size: {:.2} MB)",
                left_path
                    .clone()
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                left_size
            )));
            left_label.set_ellipsize(gtk::pango::EllipsizeMode::End);
            left_box.append(&left_label);

            let image1 = gtk::Picture::new();
            image1.set_filename(Some(left_path.clone()));
            image1.set_can_shrink(true);
            image1.set_height_request(200);
            image1.set_content_fit(gtk::ContentFit::ScaleDown);
            left_box.append(&image1);
            image_box.append(&left_box);

            // Right image with label
            let right_box = gtk::Box::new(gtk::Orientation::Vertical, 5);
            let right_size = std::fs::metadata(right_path.clone())
                .map(|m| m.len() as f64 / (1024.0 * 1024.0))
                .unwrap_or(0.0);

            let right_label = gtk::Label::new(Some(&format!(
                "Image 2: {} (Size: {:.2} MB)",
                right_path
                    .clone()
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                right_size
            )));
            right_label.set_ellipsize(gtk::pango::EllipsizeMode::End);
            right_box.append(&right_label);

            let image2 = gtk::Picture::new();
            image2.set_filename(Some(right_path.clone()));
            image2.set_can_shrink(true);
            image2.set_height_request(200);
            image2.set_content_fit(gtk::ContentFit::ScaleDown);
            right_box.append(&image2);
            image_box.append(&right_box);

            pair_box.append(&image_box);

            // Only add action buttons to the current pair
            if i == pair_index {
                // Button container
                let button_box = gtk::Box::new(gtk::Orientation::Horizontal, 10);
                button_box.set_homogeneous(true);
                button_box.set_margin_top(10);

                // Add buttons
                let keep_first = gtk::Button::with_label("Keep Image 1");
                let keep_second = gtk::Button::with_label("Keep Image 2");
                let skip = gtk::Button::with_label("Skip");
                let auto = gtk::Button::with_label("Auto Process Remaining");

                button_box.append(&keep_first);
                button_box.append(&keep_second);
                button_box.append(&skip);
                button_box.append(&auto);

                // Connect button signals
                let sender = self.sender.clone();
                keep_first.connect_clicked(clone!(@strong sender, @strong pair_index, @strong left_path, @strong right_path => move |_| {
                    let _ = sender.send(Event::KeepImage(pair_index, left_path.clone(), right_path.clone()));
                }));

                let sender = self.sender.clone();
                keep_second.connect_clicked(clone!(@strong sender, @strong pair_index, @strong left_path, @strong right_path => move |_| {
                    let _ = sender.send(Event::KeepImage(pair_index, right_path.clone(), left_path.clone()));
                }));

                let sender = self.sender.clone();
                skip.connect_clicked(clone!(@strong sender, @strong pair_index => move |_| {
                    let _ = sender.send(Event::SkipPair(pair_index));
                }));

                let sender = self.sender.clone();
                auto.connect_clicked(clone!(@strong sender => move |_| {
                    let _ = sender.send(Event::AutoProcessRemaining);
                }));

                pair_box.append(&button_box);

                // Store widgets for later access
                self.widgets.keep_first_button = Some(keep_first);
                self.widgets.keep_second_button = Some(keep_second);
                self.widgets.skip_button = Some(skip);
                self.widgets.auto_button = Some(auto);
                self.widgets.score_label = Some(score_label);
            }

            pair_frame.set_child(Some(&pair_box));
            pairs_box.append(&pair_frame);
        }

        // Add CSS for highlighting the current pair
        let provider = gtk::CssProvider::new();
        provider.load_from_data(".selected-pair { border: 2px solid #3584e4; }");

        gtk::style_context_add_provider_for_display(
            &gtk::gdk::Display::default().expect("Could not get default display"),
            &provider,
            gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );

        // Scroll to the current pair
        if let Some(frame) = pairs_box.first_child() {
            for _ in 0..pair_index {
                if let Some(next) = frame.next_sibling() {
                    // Continue to the next sibling
                    let _ = next;
                } else {
                    break;
                }
            }
            // The scrolled window will automatically scroll to show the selected pair
        }
    }
}

//-------------------------------------------------------------------------------
// UI Component Builders
//-------------------------------------------------------------------------------

fn create_directory_selector(label_text: &str, parent: &gtk::Box) -> (gtk::Entry, gtk::Button) {
    let box_container = gtk::Box::new(gtk::Orientation::Horizontal, 5);
    let label = gtk::Label::new(Some(label_text));
    box_container.append(&label);

    let path_entry = gtk::Entry::new();
    path_entry.set_hexpand(true);
    path_entry.set_placeholder_text(Some("Not selected"));
    box_container.append(&path_entry);

    let button = gtk::Button::with_label("Browse");
    box_container.append(&button);

    parent.append(&box_container);
    (path_entry, button)
}

fn create_media_type_selector(parent: &gtk::Box) -> (gtk::Switch, gtk::Label) {
    let box_container = gtk::Box::new(gtk::Orientation::Horizontal, 5);
    let label = gtk::Label::new(Some("Media Type:"));
    box_container.append(&label);

    let media_switch = gtk::Switch::new();
    media_switch.set_active(true); // Default to photos
    box_container.append(&media_switch);

    let media_type_label = gtk::Label::new(Some("Photos"));
    box_container.append(&media_type_label);

    parent.append(&box_container);
    (media_switch, media_type_label)
}

fn create_action_buttons(
    parent: &gtk::Box,
) -> (gtk::Button, gtk::Button, gtk::Button, gtk::Button) {
    let button_section = gtk::Box::new(gtk::Orientation::Vertical, 5);
    button_section.set_margin_top(10);
    button_section.set_margin_bottom(10);

    // Add a header label for the actions section
    let actions_label = gtk::Label::new(Some("Actions"));
    actions_label.set_halign(gtk::Align::Start);
    actions_label.set_margin_bottom(5);
    actions_label.set_markup("<b>Actions</b>");
    button_section.append(&actions_label);

    // Create horizontal button box for the action buttons
    let button_box = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    button_box.set_homogeneous(true); // Make buttons equal width

    // Scan button
    let scan_button = gtk::Button::with_label("Start Scan");
    button_box.append(&scan_button);

    // Add Remove Duplicates button
    let remove_duplicates_button = gtk::Button::with_label("Remove Duplicates");
    remove_duplicates_button.set_sensitive(false); // Initially disabled until scan completes
    button_box.append(&remove_duplicates_button);

    // Add Sort and Copy button
    let sort_copy_button = gtk::Button::with_label("Sort and Copy to Target");
    sort_copy_button.set_sensitive(false); // Initially disabled until scan completes
    button_box.append(&sort_copy_button);

    // Add Find Similar button
    let find_similar_button = gtk::Button::with_label("Find Similar");
    button_box.append(&find_similar_button);

    // Add the button box to the button section
    button_section.append(&button_box);

    // Add the button section to the main container
    parent.append(&button_section);

    (
        scan_button,
        remove_duplicates_button,
        sort_copy_button,
        find_similar_button,
    )
}
