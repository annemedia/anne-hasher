// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![cfg_attr(not(feature = "show_console"), windows_subsystem = "windows")]
mod cpu_hasher;
#[cfg(feature = "opencl")]
mod gpu_hasher;
#[cfg(feature = "opencl")]
mod ocl;
mod hasher;
mod poc_hashing;
mod scheduler;
mod shabal256;
mod utils;
mod writer;
mod buffer;

use std::cmp::min;
use std::env;
use std::process;

use clap::{Arg, ArgAction, ArgGroup, Command};
use hasher::{Hasher, HasherTask};

#[cfg(feature = "gui")]
use hasher::ProgressUpdate;

use utils::set_low_prio;
use utils::calculate_rounded_nonces;
// use crate::utils::{timestamp}; 
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[cfg(feature = "gui")]
mod gui_app {
    use super::*;
    use crossbeam_channel::{bounded, Receiver};
    use eframe::egui::{Color32, Context, RichText, Ui, Vec2, Widget};
    use std::path::PathBuf;
    use image::GenericImageView;
    use egui::IconData;

    #[cfg(feature = "opencl")]
    use crate::ocl::GpuInfo;

    const BG_DARK: Color32 = Color32::from_rgb(17, 17, 17);
    const TEXT_LIGHT: Color32 = Color32::from_rgb(254, 254, 254);
    const INPUT_BG: Color32 = Color32::from_rgb(254, 254, 254);
    const INPUT_TEXT: Color32 = Color32::from_rgb(17, 17, 17);
    const PRIMARY_YELLOW: Color32 = Color32::from_rgb(254, 254, 23);
    const LOG_BG: Color32 = Color32::from_rgb(0, 0, 0);
    const LOG_GREEN: Color32 = Color32::from_rgb(0, 255, 0);
    const PROGRESS_GREEN: Color32 = Color32::from_rgb(254, 23, 223);
    const WRITE_BLUE: Color32 = Color32::from_rgb(0, 153, 255);

    pub const NONCE_SIZE_BYTES: u64 = 262144;

    fn load_icon_data() -> Option<IconData> {

        let icon_bytes = include_bytes!("logo.png");

        match image::load_from_memory(icon_bytes) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let (width, height) = img.dimensions();
                
                Some(IconData {
                    rgba: rgba.into_raw(),
                    width,
                    height,
                })
            }
            Err(e) => {
                eprintln!("Failed to load icon: {}", e);
                None
            }
        }
    }

    fn display_logo(ui: &mut Ui, ctx: &Context) {
        static LOGO_BYTES: &[u8] = include_bytes!("logo.png");
        
        match image::load_from_memory(LOGO_BYTES) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let width = rgba.width() as usize;
                let height = rgba.height() as usize;
                let pixels = rgba.into_raw();
                
                let color_image = eframe::egui::ColorImage::from_rgba_unmultiplied(
                    [width, height],
                    &pixels
                );
                
                let texture = ctx.load_texture(
                    "anne-logo",
                    color_image,
                    eframe::egui::TextureOptions::default()
                );
                
                eframe::egui::Image::new(&texture)
                    .fit_to_exact_size(eframe::egui::Vec2::new(40.0, 40.0))
                    .ui(ui);
            }
            Err(_e) => {

                let (rect, _response) = ui.allocate_exact_size(Vec2::new(40.0, 40.0), eframe::egui::Sense::hover());
                

                ui.painter().rect_filled(
                    rect,
                    eframe::egui::CornerRadius::same(8),
                    PRIMARY_YELLOW,
                );
                

                let text = eframe::egui::RichText::new("AP").color(BG_DARK).size(20.0);
                

                let job = eframe::egui::text::LayoutJob::simple(
                    text.text().to_string(),
                    eframe::egui::TextStyle::Heading.resolve(ui.style()),
                    BG_DARK,
                    f32::INFINITY,
                );
                
                let galley = ui.painter().layout_job(job);
                let text_pos = rect.center() - galley.size() / 2.0;
                ui.painter().galley(text_pos, galley, BG_DARK);
            }
        }
    }

pub fn launch() -> eframe::Result<()> {
    println!("Launching GUI...");
    
    let icon_data = load_icon_data();
    println!("Icon data loaded: {}", icon_data.is_some());
    
    let mut options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([720.0, 800.0])
            .with_resizable(true)
            .with_title("ANNE Hasher"),
        ..Default::default()
    };

    if let Some(icon) = icon_data {
        println!("Setting window icon...");
        options.viewport = options.viewport.with_icon(std::sync::Arc::new(icon));
    } else {
        println!("Warning: Could not load icon data");
    }

    eframe::run_native(
        "ANNE Hasher",
        options,
        Box::new(|_cc| {
            println!("Creating app instance...");
            Ok(Box::new(AnneGuiApp::default()))
        }),
    )
}

struct AnneGuiApp {
    numeric_id: String,
    start_nonce: String,
    auto_mode: bool,
    auto_count: String,
    nonces: String,
    total_nonces: u64,
    path: PathBuf,
    cpu_cores: String,
    #[cfg(feature = "opencl")]
    detected_gpus: Vec<GpuInfo>,
    #[cfg(feature = "opencl")]
    selected_gpu: usize,
    #[cfg(feature = "opencl")]
    gpu_cores: String,
    #[cfg(feature = "opencl")]
    total_gpu_cores: u32,
    #[cfg(feature = "opencl")]
    gpu_detection_done: bool,
    #[cfg(feature = "opencl")]
    gpu_rx: Option<Receiver<Vec<GpuInfo>>>,
    #[cfg(feature = "opencl")]
    gpu_thread_spawned: bool,   // ← Add this line
    visuals_applied: bool,
    disable_direct_io: bool,
    low_priority: bool,
    benchmark: bool,
    #[cfg(feature = "opencl")]
    zero_copy: bool,
    progress: f32,
    write_progress: f32,
    speed: f64,
    write_speed: f64,
    eta: String,
    logs: Vec<String>,
    running: bool,
    stop_requested: bool,
    stop_flag: Option<Arc<AtomicBool>>,
    error: Option<String>,
    rx: Option<Receiver<ProgressUpdate>>,
}

impl Default for AnneGuiApp {
    fn default() -> Self {
        let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let default_nonces = 381500u64;
        let cores = sys_info::cpu_num().unwrap_or(1) / 2;

        // Define all GPU-related fields (even if not using OpenCL)
        #[cfg(feature = "opencl")]
        let (gpu_cores, total_gpu_cores, detected_gpus, selected_gpu, gpu_detection_done, gpu_rx, gpu_thread_spawned) = (
            "0".to_string(),
            0u32,
            Vec::new(),
            0usize,
            false,
            None,
            false,
        );

        #[cfg(not(feature = "opencl"))]
        let (gpu_cores, total_gpu_cores, detected_gpus, selected_gpu, gpu_detection_done, gpu_rx, gpu_thread_spawned) = (
            "0".to_string(),
            0u32,
            Vec::new(),
            0usize,
            false,
            None,
            false,
        );

        Self {
             #[cfg(feature = "gui")]
            numeric_id: "".to_string(),
            start_nonce: "0".to_string(),
            auto_mode: true,
            auto_count: "1".to_string(),
            nonces: default_nonces.to_string(),
            total_nonces: default_nonces,
            path: PathBuf::from(home),
            cpu_cores: cores.to_string(),
            #[cfg(feature = "opencl")]
            detected_gpus,
            #[cfg(feature = "opencl")]
            selected_gpu,
            #[cfg(feature = "opencl")]
            gpu_cores,
            #[cfg(feature = "opencl")]
            total_gpu_cores,
            #[cfg(feature = "opencl")]
            gpu_detection_done,
            #[cfg(feature = "opencl")]
            gpu_rx,
            #[cfg(feature = "opencl")]
            gpu_thread_spawned,
            visuals_applied: false,
            disable_direct_io: false,
            low_priority: false,
            benchmark: false,
            #[cfg(feature = "opencl")]
            zero_copy: false,
            progress: 0.0,
            write_progress: 0.0,
            speed: 0.0,
            write_speed: 0.0,
            eta: "Unknown".to_string(),
            logs: vec!["ANNE Hasher ready — the ultimate PoST hasher.".to_string()],
            running: false,
            stop_requested: false,
            stop_flag: None,
            error: None,
            rx: None,
        }
    }
}

    impl eframe::App for AnneGuiApp {
        fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
            #[cfg(feature = "opencl")]
            self.poll_gpu_detection();

            if !self.visuals_applied {
                let mut visuals = eframe::egui::Visuals::dark();
                visuals.panel_fill = BG_DARK;
                visuals.extreme_bg_color = BG_DARK;
                visuals.faint_bg_color = Color32::from_rgb(30, 30, 30);
                visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(35, 35, 35);
                visuals.widgets.inactive.bg_fill = Color32::from_rgb(45, 45, 45);
                visuals.widgets.hovered.bg_fill = Color32::from_rgb(60, 60, 60);
                visuals.widgets.active.bg_fill = PRIMARY_YELLOW.linear_multiply(0.2);
                visuals.selection.bg_fill = PRIMARY_YELLOW.linear_multiply(0.3);
                visuals.selection.stroke.color = PRIMARY_YELLOW;

                ctx.set_visuals(visuals);
                self.visuals_applied = true;
            }

            self.poll_progress();
            self.update_nonces_size();

            // Use a scroll area for the entire content
            eframe::egui::CentralPanel::default()
                .show(ctx, |ui| {
                    // Add a vertical scroll area that covers the entire panel
                    eframe::egui::ScrollArea::vertical()
                        .auto_shrink([false; 2]) // Don't shrink when content fits
                        .show(ui, |ui| {
                            // Main content container with padding
                            ui.vertical(|ui| {
                                // Header with logo
                                ui.horizontal(|ui| {
                                    display_logo(ui, ctx);
                                    ui.add_space(5.0);
                                    ui.vertical(|ui| {
                                        ui.add_space(2.0);
                                        ui.label(RichText::new("ANNE Hasher").size(17.0).color(PRIMARY_YELLOW));
                                        ui.label(RichText::new("Proof of Spacetime Hasher").size(12.0).color(TEXT_LIGHT));
                                    });
                                    ui.with_layout(eframe::egui::Layout::right_to_left(eframe::egui::Align::Center), |ui| {
                                        if self.running {
                                            if ui.add(eframe::egui::Button::new(RichText::new("STOP").size(14.0).color(BG_DARK))
                                                .fill(Color32::from_rgb(255, 100, 100))
                                                .min_size(Vec2::new(100.0, 35.0)))
                                                .clicked()
                                            {
                                                self.stop_hashing();
                                            }
                                            ui.add_space(10.0);
                                            ui.spinner();
                                            ui.label(RichText::new("Running...").color(PRIMARY_YELLOW).size(14.0));
                                        } else {
                                            if ui.add(eframe::egui::Button::new(RichText::new("START").size(14.0).color(BG_DARK))
                                                .fill(PRIMARY_YELLOW)
                                                .min_size(Vec2::new(100.0, 35.0)))
                                                .clicked()
                                            {
                                                self.start_hashing();
                                            }
                                        }
                                    });
                                });
                            
                                ui.add_space(15.0);
                   
                                // Input section
                                ui.group(|ui| {
                                    self.render_inputs(ui);
                                });
                                
                                ui.add_space(10.0);
                                
                                // Status section
                                ui.group(|ui| {
                                    self.render_status(ui);
                                });
                                
                                ui.add_space(10.0);
                                
                                // Logs section
                                ui.group(|ui| {
                                    self.render_logs(ui);
                                });
                                
                                // Add some bottom padding
                                ui.add_space(20.0);
                            });
                        });
                });
            
            // Modal dialogs (outside the scroll area)
            self.show_stop_confirmation(ctx);     
            self.show_error_popup(ctx);
            
            if self.running || self.progress < 1.0 || self.write_progress < 1.0 {
                ctx.request_repaint_after(std::time::Duration::from_millis(300));
            }
        }
    }

    impl AnneGuiApp {
        #[cfg(feature = "opencl")]
        fn poll_gpu_detection(&mut self) {
            if !self.gpu_detection_done {
                if !self.gpu_thread_spawned {
                    let (tx, rx) = bounded::<Vec<GpuInfo>>(1);
                    self.gpu_rx = Some(rx);
                    self.gpu_thread_spawned = true;
                    std::thread::spawn(move || {
                        let gpus = crate::ocl::get_gpu_list();
                        let _ = tx.send(gpus);
                    });
                }
                if let Some(rx) = &self.gpu_rx {
                    match rx.try_recv() {
                        Ok(gpus) => {
                            self.detected_gpus = gpus;
                            self.gpu_detection_done = true;
                            self.gpu_rx = None;
                            self.selected_gpu = if self.detected_gpus.is_empty() { 0 } else { 1 };
                            let (gpu_cores, total_gpu_cores) = if self.selected_gpu > 0 {
                                let idx = self.selected_gpu - 1;
                                if idx < self.detected_gpus.len() {
                                    let gpu = &self.detected_gpus[idx];
                                    let parts: Vec<&str> = gpu.spec.split(':').collect();
                                    if parts.len() >= 3 {
                                        if let Ok(total_cores) = parts[2].parse::<u32>() {
                                            let quarter_cores = (total_cores as f32 * 0.25).ceil() as u32;
                                            (quarter_cores.to_string(), total_cores)
                                        } else {
                                            ("0".to_string(), 0)
                                        }
                                    } else {
                                        ("0".to_string(), 0)
                                    }
                                } else {
                                    ("0".to_string(), 0)
                                }
                            } else {
                                ("0".to_string(), 0)
                            };
                            self.gpu_cores = gpu_cores;
                            self.total_gpu_cores = total_gpu_cores;
                            self.logs.push(if self.detected_gpus.is_empty() {
                                "No GPUs detected. Using CPU only.".to_string()
                            } else {
                                format!("Detected {} GPU(s).", self.detected_gpus.len())
                            });
                        }
                        Err(crossbeam_channel::TryRecvError::Disconnected) => {
                            self.gpu_detection_done = true;
                            self.gpu_rx = None;
                            self.logs.push("GPU detection failed.".to_string());
                        }
                        Err(crossbeam_channel::TryRecvError::Empty) => {
                            // Still waiting...
                        }
                    }
                }
            }
        }

        fn update_nonces_size(&mut self) {
            if let Ok(n) = self.nonces.parse::<u64>() {
                self.total_nonces = n;
            }
        }

        fn render_inputs(&mut self, ui: &mut Ui) {
            let available_width = ui.available_width() - 20.0;
            

            ui.horizontal(|ui| {
                 ui.add_space(5.0);
                ui.vertical(|ui| {
                    ui.heading(RichText::new("Basic Hashing Parameters").color(PRIMARY_YELLOW).size(14.0));
                    ui.add_space(5.0);
                    
                    ui.group(|ui| {
                        ui.vertical(|ui| {

                            ui.horizontal(|ui| {
                                ui.vertical(|ui| {
                                    ui.label(RichText::new("ANNE ID:").color(TEXT_LIGHT));
                                    ui.add(
                                        eframe::egui::TextEdit::singleline(&mut self.numeric_id)
                                            .desired_width((available_width / 2.0) - 20.0)
                                            .frame(true)
                                            .background_color(INPUT_BG)
                                            .text_color(INPUT_TEXT)
                                    );
                                });
                                
                                ui.add_space(10.0);
                                
                                ui.vertical(|ui| {
                                    let total_bytes = self.total_nonces as f64 * NONCE_SIZE_BYTES as f64;
                                    let display_size = if total_bytes >= 1e12 {
                                        format!("{:.2} TB", total_bytes / 1e12)
                                    } else if total_bytes >= 1e9 {
                                        let gb = total_bytes / 1e9;
                                        if gb < 10.0 {
                                            format!("{:.2} GB", gb)
                                        } else {
                                            format!("{:.1} GB", gb)
                                        }
                                    } else {
                                        format!("{:.1} MB", total_bytes / 1e6)
                                    };

                                    ui.label(RichText::new(format!("Nonces per file (~{}):", display_size)).color(TEXT_LIGHT));
                                    ui.add(
                                        eframe::egui::TextEdit::singleline(&mut self.nonces)
                                            .desired_width((available_width / 2.0) - 20.0)
                                            .frame(true)
                                            .background_color(INPUT_BG)
                                            .text_color(INPUT_TEXT)
                                    );
                                });
                            });

                            ui.add_space(15.0);

                            ui.horizontal(|ui| {
                                ui.vertical(|ui| {
                                    ui.label(RichText::new(if self.auto_mode { "Number of files:" } else { "Start Nonce:" }).color(TEXT_LIGHT));
                                    ui.add(
                                        eframe::egui::TextEdit::singleline(if self.auto_mode { &mut self.auto_count } else { &mut self.start_nonce })
                                            .desired_width((available_width / 2.0) - 20.0)
                                            .frame(true)
                                            .background_color(INPUT_BG)
                                            .text_color(INPUT_TEXT)
                                    );
                                    ui.checkbox(&mut self.auto_mode, RichText::new("Auto-hashing sequential files").color(TEXT_LIGHT));
                                });
                                
                                ui.add_space(10.0);
                                
                                ui.vertical(|ui| {
                                    ui.label(RichText::new("Output Directory:").color(TEXT_LIGHT));
                                    ui.horizontal(|ui| {
                                        
                                        if ui.add(eframe::egui::Button::new(RichText::new("Browse...").color(PRIMARY_YELLOW))
                                            .fill(Color32::from_rgb(45, 45, 45))
                                            .min_size(Vec2::new(80.0, 30.0)))
                                            .clicked() 
                                        {
                                            if let Some(p) = rfd::FileDialog::new().pick_folder() {
                                                self.path = p;
                                            }
                                        }
                                        ui.label(RichText::new(self.path.to_string_lossy()).color(TEXT_LIGHT).weak());
                                    });
                                });
                            });
                        });
                    });
                });
             });

            ui.add_space(10.0);

            ui.horizontal(|ui| {
                ui.add_space(5.0);
                ui.vertical(|ui| {
                    ui.heading(RichText::new("Processing Configuration").color(PRIMARY_YELLOW).size(14.0));
                    ui.add_space(5.0);
                    
                    ui.group(|ui| {
                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                ui.vertical(|ui| {

                                    let total_cpu_cores = sys_info::cpu_num().unwrap_or(1);
                                    let cpu_label = format!("CPU Cores (Total: {}, 0 = none):", total_cpu_cores);
                                    
                                    ui.label(RichText::new(cpu_label).color(TEXT_LIGHT));
                                    ui.add(
                                        eframe::egui::TextEdit::singleline(&mut self.cpu_cores)
                                            .desired_width((available_width / 2.0) - 20.0)
                                            .frame(true)
                                            .background_color(INPUT_BG)
                                            .text_color(INPUT_TEXT)
                                    );
                                });
                                
                                ui.add_space(10.0);
                                                    
                                ui.vertical(|ui| {

                                    let total_label = {
                                        #[cfg(feature = "opencl")]
                                        {
                                            if self.selected_gpu > 0 && self.total_gpu_cores > 0 {
                                                format!("GPU Cores (Total: {}):", self.total_gpu_cores)
                                            } else {
                                                "GPU Cores:".to_string()
                                            }
                                        }
                                        #[cfg(not(feature = "opencl"))]
                                        {
                                            "GPU Cores:".to_string()
                                        }
                                    };
                                    
                                    ui.label(RichText::new(total_label).color(TEXT_LIGHT));
                                    ui.add(
                                        eframe::egui::TextEdit::singleline(&mut self.gpu_cores)
                                            .desired_width((available_width / 2.0) - 20.0)
                                            .frame(true)
                                            .background_color(INPUT_BG)
                                            .text_color(INPUT_TEXT)
                                    );
                                });
                            });
                            
                            ui.add_space(10.0);
                            
                            #[cfg(feature = "opencl")]
                            {
                                if !self.gpu_detection_done {
                                    ui.label(RichText::new("Detecting GPUs...").color(TEXT_LIGHT));
                                } else {
                                    ui.label(RichText::new("Select GPU:").color(TEXT_LIGHT));
                                    
                                    let mut gpu_options = vec!["No GPU".to_string()];
                                    for gpu in &self.detected_gpus {
                                        gpu_options.push(format!("{} - {}", gpu.vendor, gpu.name));
                                    }
                                    
                                    eframe::egui::ComboBox::from_id_salt("gpu_select")
                                        .selected_text(gpu_options[self.selected_gpu].clone())
                                        .width(available_width - 10.0)
                                        .show_ui(ui, |ui| {
                                            for (idx, option) in gpu_options.iter().enumerate() {
                                                let response = ui.selectable_value(&mut self.selected_gpu, idx, option);
                                                

                                                if response.changed() {
                                                    if idx == 0 {

                                                        self.total_gpu_cores = 0;
                                                        self.gpu_cores = "0".to_string();
                                                    } else {

                                                        let gpu_idx = idx - 1;
                                                        if gpu_idx < self.detected_gpus.len() {
                                                            let gpu = &self.detected_gpus[gpu_idx];
                                                            let parts: Vec<&str> = gpu.spec.split(':').collect();
                                                            if parts.len() >= 3 {
                                                                if let Ok(total_cores) = parts[2].parse::<u32>() {

                                                                    self.total_gpu_cores = total_cores;

                                                                    let quarter_cores = (total_cores as f32 * 0.25).ceil() as u32;
                                                                    self.gpu_cores = quarter_cores.to_string();
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        });
                                }
                            }
                            
                            #[cfg(not(feature = "opencl"))]
                            {
                                ui.label(RichText::new("GPU support not compiled in.").color(TEXT_LIGHT));
                            }
                        });
                    });
                });
            });

            ui.add_space(10.0);

            ui.horizontal(|ui| {
                ui.add_space(5.0);
                ui.vertical(|ui| {
                    ui.heading(RichText::new("Options").color(PRIMARY_YELLOW).size(14.0));
                    ui.add_space(5.0);
                    
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing = Vec2::new(40.0, 10.0);
                            
                            ui.horizontal(|ui| {
                                let response = ui.checkbox(&mut self.benchmark, RichText::new("Benchmark Mode").color(TEXT_LIGHT));
                                response.on_hover_text("Measure hashing performance without writing nonces to disk.");
                                
                                let response = ui.checkbox(&mut self.low_priority, RichText::new("Low Priority").color(TEXT_LIGHT));
                                response.on_hover_text("Reduce process priority to minimize impact on system performance.");
                                
                                let response = ui.checkbox(&mut self.disable_direct_io, RichText::new("Disable Direct I/O").color(TEXT_LIGHT));
                                response.on_hover_text("Use buffered I/O instead of direct disk access - typically needed on LUKS on top of dm-crypt with LVM for the root filesystem");
                                #[cfg(feature = "opencl")]
                                let response = ui.checkbox(&mut self.zero_copy, RichText::new("Zero-Copy Buffers (iGPU)").color(TEXT_LIGHT));
                                #[cfg(feature = "opencl")]
                                response.on_hover_text("Enable zero-copy memory transfers for integrated GPUs. Intel GPUs are automatically detected and handled with fallback.");
                            });
                        });
                    });
                });
            });
        }



        fn render_status(&mut self, ui: &mut Ui) {
            ui.horizontal(|ui| {
                ui.add_space(5.0);
                ui.vertical(|ui| {
                    ui.heading(RichText::new("Progress").size(14.0).color(PRIMARY_YELLOW));
                    ui.add_space(5.0);
                    
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Hashing:").color(TEXT_LIGHT).size(14.0));
                        ui.add_space(10.0);
                        let hashing_bar = eframe::egui::ProgressBar::new(self.progress)
                            .desired_width(ui.available_width() - 6.0)
                            .animate(self.running)
                            .fill(PROGRESS_GREEN)
                            .text(RichText::new(format!("{:.1}%", self.progress * 100.0)).color(TEXT_LIGHT));
                        ui.add(hashing_bar);
                    });
                    
                    ui.add_space(8.0);
                    
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Writing:").color(TEXT_LIGHT).size(14.0));
                        ui.add_space(15.0);
                        let writing_bar = eframe::egui::ProgressBar::new(self.write_progress)
                            .desired_width(ui.available_width() - 6.0)
                            .animate(self.running)
                            .fill(WRITE_BLUE)
                            .text(RichText::new(format!("{:.1}%", self.write_progress * 100.0)).color(TEXT_LIGHT));
                        ui.add(writing_bar);
                    });

                    ui.add_space(10.0);
                    
                    ui.horizontal(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(format!("Hashing Speed: {:.1} nonces/sec", self.speed)).color(TEXT_LIGHT));
                            ui.add_space(20.0);
                            ui.label(RichText::new(format!("Write Speed: {:.1} MB/sec", self.write_speed)).color(TEXT_LIGHT));
                            ui.add_space(20.0);
                            ui.label(RichText::new(format!("ETA: {}", self.eta)).color(TEXT_LIGHT));
                        });
                    });
                });
            });
        }

        fn render_logs(&mut self, ui: &mut Ui) {
            ui.horizontal(|ui| {
                ui.add_space(5.0);
                ui.vertical(|ui| {

                    ui.horizontal(|ui| {
                        ui.add_space(5.0);
                        ui.heading(RichText::new("Logs").color(PRIMARY_YELLOW).size(13.0));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Clear").clicked() {
                                self.logs.clear();
                            }
                            ui.add_space(5.0);
                            if ui.button("Copy").clicked() {
                                ui.ctx().copy_text(self.logs.join("\n"));
                            }
                        });
                    });
                    
                    ui.add_space(5.0);
                    

                    let full_width = ui.available_width() - 30.0;
                    

                    egui::Frame::new()
                        .fill(LOG_BG)
                        .inner_margin(10.0)
                        .show(ui, |ui| {

                            ui.set_width(full_width);
                            

                            let log_height = 140.0;
                            ui.set_min_height(log_height);
                            

                            eframe::egui::ScrollArea::vertical()
                                .max_width(full_width)
                                .stick_to_bottom(true)
                                .show(ui, |ui| {

                                    ui.set_width(full_width);
                                    
                                    for line in &self.logs {
                                        ui.label(
                                            RichText::new(line)
                                                .color(LOG_GREEN)
                                                .monospace()
                                                .background_color(LOG_BG)
                                        );
                                    }
                                });
                        });
                });
            });
        }
        fn show_error_popup(&mut self, ctx: &Context) {
            let err = self.error.clone();
            if let Some(err) = &err {
                eframe::egui::Window::new("Error")
                    .anchor(eframe::egui::Align2::CENTER_CENTER, Vec2::ZERO)
                    .fixed_size([500.0, 200.0])
                    .show(ctx, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.add_space(20.0);
                            ui.label(RichText::new(err).color(Color32::from_rgb(255, 100, 100)));
                            ui.add_space(30.0);
                            ui.horizontal(|ui| {
                                if ui.add(eframe::egui::Button::new("OK").min_size(Vec2::new(100.0, 40.0))).clicked() {
                                    self.error = None;
                                }
                            });
                        });
                    });
            }
        }
    
        fn delete_partial_file(&mut self) {
            use std::fs;
            
            // Parse numeric ID
            if let Ok(numeric_id) = self.numeric_id.trim().parse::<u64>() {
                // Look for ANY file with this numeric ID
                if let Ok(entries) = fs::read_dir(&self.path) {
                    let mut deleted_any = false;
                    for entry in entries.flatten() {
                        if let Some(file_name) = entry.file_name().to_str() {
                            if file_name.starts_with(&format!("{}_", numeric_id)) {
                                let path = entry.path();
                                if path.is_file() {
                                    match fs::remove_file(&path) {
                                        Ok(_) => {
                                            self.logs.push(format!("Deleted: {}", file_name));
                                            deleted_any = true;
                                        }
                                        Err(e) => {
                                            self.logs.push(format!("Failed to delete {}: {}", file_name, e));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if !deleted_any {
                        self.logs.push("No hash files found to delete.".to_string());
                    }
                }
            } else {
                self.logs.push("Invalid numeric ID - cannot delete files.".to_string());
            }
        }
        
        fn stop_hashing(&mut self) {
            if self.running {
                self.stop_requested = true;  // Just request stop, don't set flag yet
                self.logs.push("Stop requested. Waiting for confirmation...".to_string());
                
                // DON'T set stop_flag here yet - wait for confirmation
                // DON'T delete file yet - wait for confirmation
            }
        }

        fn confirm_stop(&mut self) {
            if let Some(flag) = &self.stop_flag {
                flag.store(true, Ordering::Relaxed);
            }
            
            self.delete_partial_file();
            
            self.running = false;
            self.stop_requested = false;
            
            self.logs.push("Stop confirmed. Stopping hasher...".to_string());
        }
        
        fn show_stop_confirmation(&mut self, ctx: &Context) {
            if self.stop_requested && self.running {
                // Simple backdrop
                egui::Area::new("modal_back".into())
                    .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                    .interactable(true)
                    .show(ctx, |ui| {
                        let rect = ui.max_rect();
                        ui.painter().rect_filled(rect, egui::CornerRadius::ZERO, 
                            Color32::from_black_alpha(120));
                    });

                egui::Window::new("")
                    .collapsible(false)
                    .resizable(false)
                    .title_bar(false)
                    .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                    .frame(egui::Frame::window(&ctx.style())
                        .fill(Color32::from_rgb(35, 35, 40))
                        .corner_radius(10.0))
                    .fixed_size(egui::Vec2::new(440.0, 230.0))
                    .show(ctx, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.add_space(30.0);
                            
                            ui.label(
                                egui::RichText::new("⚠ Stop Hashing?")
                                    .color(Color32::from_rgb(255, 180, 50))
                                    .size(22.0)
                            );
                            
                            ui.add_space(15.0);
                            
                            ui.label("The partial file will be permanently deleted.");
                            
                            ui.add_space(40.0);
                            
                            ui.horizontal(|ui| {
                                ui.add_space(20.0);
                                
                                // Cancel button with nice styling
                                let cancel_btn = egui::Button::new(
                                    egui::RichText::new("CANCEL")
                                        .color(Color32::from_rgb(230, 230, 230))
                                )
                                .fill(Color32::from_rgb(60, 60, 70))
                                .min_size(egui::Vec2::new(120.0, 40.0));
                                
                                if ui.add(cancel_btn).clicked() {
                                    self.stop_requested = false;
                                }
                                
                                ui.add_space(20.0);
                                
                                // Stop button - bold and attention-grabbing
                                let stop_btn = egui::Button::new(
                                    egui::RichText::new("STOP")
                                        .color(Color32::WHITE)
                                        .strong()
                                )
                                .fill(Color32::from_rgb(200, 60, 60))
                                .min_size(egui::Vec2::new(120.0, 40.0));
                                
                                if ui.add(stop_btn).clicked() {
                                    self.confirm_stop();
                                }
                                
                                ui.add_space(20.0);
                            });
                        });
                    });
            }
        }

        fn poll_progress(&mut self) {
            if let Some(rx) = &self.rx {
                while let Ok(update) = rx.try_recv() {
                    match update {
                        ProgressUpdate::Log(msg) => {
                            self.logs.push(msg);
                            if self.logs.len() > 1000 {
                                self.logs.drain(0..500);
                            }
                        }
                        ProgressUpdate::Progress(p) => {
                            self.progress = p;
                        }
                        ProgressUpdate::WriteProgress(wp) => {
                            self.write_progress = wp;
                        }
                        ProgressUpdate::Speed(s) => {
                            let per_sec = s / 60.0;
                            self.speed = per_sec;
                            if self.total_nonces > 0 && per_sec > 0.0 {
                                let remaining_nonces = self.total_nonces as f64 * (1.0 - self.progress as f64);
                                let remaining_sec = remaining_nonces / per_sec;
                                self.eta = if remaining_sec > 3600.0 {
                                    format!("{:.1}h", remaining_sec / 3600.0)
                                } else if remaining_sec > 60.0 {
                                    format!("{:.1}m", remaining_sec / 60.0)
                                } else {
                                    format!("{}s", remaining_sec as u64)
                                };
                            } else {
                                self.eta = "Unknown".to_string();
                            }
                        }
                        ProgressUpdate::WriteSpeed(ws) => {
                            self.write_speed = ws;
                        }
                        ProgressUpdate::Error(e) => {
                            if e == "STOP_REQUESTED" {
                                // This is our stop signal
                                self.running = false;
                                self.stop_flag = None;
                                // Change this message
                                self.logs.push("Hashing stopped by user.".to_string());
                            } else if e == "HASHING_INTERRUPTED" {
                                // Add a new error type for interrupted hashing
                                self.running = false;
                                self.stop_flag = None;
                                self.logs.push("Hashing interrupted.".to_string());
                            } else {
                                // Real error
                                self.error = Some(e);
                                self.running = false;
                                self.stop_flag = None;
                            }
                        }
                        ProgressUpdate::Done => {
                            self.running = false;
                            self.stop_flag = None;
                            // Only show "Hashing completed!" if actually completed
                            if !self.stop_requested {
                                self.logs.push("Hashing completed!".to_string());
                            }
                            self.progress = 1.0;
                            self.write_progress = 1.0;
                        }
                    }
                }
            }
        }
            

        fn start_hashing(&mut self) {
            use std::thread;
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::sync::Arc;

            let numeric_id: u64 = match self.numeric_id.trim().parse() {
                Ok(v) => v,
                Err(_) => {
                    self.error = Some("Invalid Numeric Account ID".to_string());
                    return;
                }
            };

            let nonces: u64 = match self.nonces.trim().parse() {
                Ok(v) => v,
                Err(_) => {
                    self.error = Some("Invalid Nonces value".to_string());
                    return;
                }
            };
            self.total_nonces = nonces;

            let cpu_threads: u8 = match self.cpu_cores.trim().parse() {
                Ok(v) => v,
                Err(_) => {
                    self.error = Some("Invalid CPU cores value".to_string());
                    return;
                }
            };

            let total_cpu_cores = sys_info::cpu_num().unwrap_or(1) as u8;

            if cpu_threads > total_cpu_cores {
                self.error = Some(format!(
                    "Cannot use {} CPU cores - system only has {}",
                    cpu_threads, total_cpu_cores
                ));
                return;
            }

            let gpu_cores_count: u32 = match self.gpu_cores.trim().parse() {
                Ok(mut v) => {
                    #[cfg(feature = "opencl")]
                    {
                        if v > self.total_gpu_cores && self.total_gpu_cores > 0 {
                            v = self.total_gpu_cores;
                            self.gpu_cores = v.to_string();
                        }
                    }
                    v
                },
                Err(_) => {
                    self.error = Some("Invalid GPU cores value".to_string());
                    return;
                }
            };

            let cpu_enabled = cpu_threads > 0;
            let gpu_enabled = {
                #[cfg(feature = "opencl")]
                {
                    self.selected_gpu > 0 && gpu_cores_count > 0 && self.gpu_detection_done
                }
                #[cfg(not(feature = "opencl"))]
                {
                    false
                }
            };
            
            if !cpu_enabled && !gpu_enabled {
                self.error = Some(
                    "Cannot start hashing: At least CPU or GPU must be enabled.\n\
                    - Set CPU cores > 0, or\n\
                    - Select a GPU and set GPU cores > 0"
                    .to_string()
                );
                return;
            }

            let mut gpus = Vec::new();
            
            #[cfg(feature = "opencl")]
            {
                if self.selected_gpu > 0 && self.gpu_detection_done {
                    let idx = self.selected_gpu - 1;
                    if idx < self.detected_gpus.len() {
                        let spec = &self.detected_gpus[idx].spec;
                        let parts: Vec<&str> = spec.split(':').collect();
                        if parts.len() >= 2 {
                            let new_spec = format!("{}:{}:{}", parts[0], parts[1], gpu_cores_count);
                            gpus.push(new_spec);
                        }
                    }
                }
            }

            let gpus = if gpus.is_empty() { None } else { Some(gpus) };

            let start_nonce: u64 = if self.auto_mode {
                let _count: u64 = match self.auto_count.trim().parse() {
                    Ok(v) if v >= 1 => v,
                    _ => {
                        self.error = Some("Auto-hashing count must be >= 1".to_string());
                        return;
                    }
                };
                0
            } else {
                match self.start_nonce.trim().parse() {
                    Ok(v) => v,
                    Err(_) => {
                        self.error = Some("Invalid Start Nonce".to_string());
                        return;
                    }
                }
            };

            if self.low_priority {
                set_low_prio();
            }

            let output_path = self.path.to_string_lossy().to_string();
            let memory = "0B".to_string();

            let direct_io = !self.disable_direct_io;
            let benchmark = self.benchmark;
            #[cfg(feature = "opencl")]
            let zero_copy = self.zero_copy;

            let (tx, rx) = bounded(5000);
            self.rx = Some(rx);
            self.progress = 0.0;
            self.write_progress = 0.0;
            self.speed = 0.0;
            self.write_speed = 0.0;
            self.eta = "Calculating...".to_string();
            self.logs.clear();
            self.logs.push("Starting hasher...".to_string());
            self.logs.push(format!("Using {} CPU cores", cpu_threads));
            if let Some(_gpu_list) = &gpus {
                self.logs.push(format!("Using GPU with {} cores", gpu_cores_count));
            } else {
                self.logs.push("Using CPU only".to_string());
            }
            self.running = true;
            self.error = None;
            self.stop_requested = false;

            // CREATE STOP FLAG HERE
            let stop_flag = Arc::new(AtomicBool::new(false));
            self.stop_flag = Some(stop_flag.clone());

            let hasher = Hasher::new();

            // Calculate rounded nonces
            let rounded_nonces = if direct_io {
                crate::utils::calculate_rounded_nonces(nonces, true, &output_path)
            } else {
                nonces
            };

            // Log if rounding happened
            if rounded_nonces != nonces {
                let _ = tx.send(ProgressUpdate::Log(
                    format!("Rounded nonces to {} for sector alignment (original: {})", rounded_nonces, nonces)
                ));
            }

            if self.auto_mode {
                let count: u64 = self.auto_count.parse().unwrap_or(1);
                
                // Clone values for the thread
                let tx_clone = tx.clone();
                let output_path_clone = output_path.clone();
                let memory_clone = memory.clone();
                let gpus_clone = gpus.clone();
                let stop_flag_clone = stop_flag.clone();
                
                thread::spawn(move || {
                    let mut current_start = 0u64;
                    if let Ok(entries) = std::fs::read_dir(&output_path_clone) {
                        let mut max_end: u64 = 0;
                        let prefix = format!("{}_", numeric_id);

                        for entry in entries.flatten() {
                            if let Some(file_name) = entry.file_name().to_str() {
                                if file_name.starts_with(&prefix) {
                                    let parts: Vec<&str> = file_name.split('_').collect();
                                    if parts.len() >= 3 {
                                        if let (Ok(sn), Ok(cnt)) = (parts[1].parse::<u64>(), parts[2].parse::<u64>()) {
                                            let end = sn + cnt;
                                            if end > max_end {
                                                max_end = end;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        current_start = max_end;
                    }

                    let _ = tx_clone.send(ProgressUpdate::Log(
                        format!("Auto-hashing {} files starting from nonce {}", count, current_start)
                    ));

                    for i in 0..count {
                        let current_file = i + 1;
                        let file_start_progress = i as f32 / count as f32;
                        
                        // Check stop flag before starting each file
                        if stop_flag_clone.load(Ordering::Relaxed) {
                            let _ = tx_clone.send(ProgressUpdate::Log(format!("Stop requested before file {} of {}", current_file, count)));
                            let _ = tx_clone.send(ProgressUpdate::Error("STOP_REQUESTED".to_string()));
                            break;
                        }
                        
                        // Send initial progress for this file
                        let _ = tx_clone.send(ProgressUpdate::Progress(file_start_progress));
                        
                        // Create a wrapper for this file's progress
                        let (file_tx, file_rx) = bounded(5000);
                        let main_tx = tx_clone.clone();
                        
                        // Thread to scale progress for this file
                        let progress_scaler = thread::spawn(move || {
                            let file_weight = 1.0 / count as f32;
                            
                            while let Ok(update) = file_rx.recv() {
                                match update {
                                    ProgressUpdate::Progress(p) => {
                                        // Scale this file's progress to overall progress
                                        let scaled_progress = file_start_progress + (p * file_weight);
                                        let _ = main_tx.send(ProgressUpdate::Progress(scaled_progress));
                                    }
                                    ProgressUpdate::WriteProgress(wp) => {
                                        // Scale write progress similarly
                                        let scaled_write_progress = file_start_progress + (wp * file_weight);
                                        let _ = main_tx.send(ProgressUpdate::WriteProgress(scaled_write_progress));
                                    }
                                    ProgressUpdate::Log(msg) => {
                                        let _ = main_tx.send(ProgressUpdate::Log(msg));
                                    }
                                    ProgressUpdate::Speed(s) => {
                                        let _ = main_tx.send(ProgressUpdate::Speed(s));
                                    }
                                    ProgressUpdate::WriteSpeed(ws) => {
                                        let _ = main_tx.send(ProgressUpdate::WriteSpeed(ws));
                                    }
                                    ProgressUpdate::Error(e) => {
                                        let _ = main_tx.send(ProgressUpdate::Error(e));
                                    }
                                    ProgressUpdate::Done => {
                                        // File is done - send completion progress
                                        let file_complete_progress = (i + 1) as f32 / count as f32;
                                        let _ = main_tx.send(ProgressUpdate::Progress(file_complete_progress));
                                        let _ = main_tx.send(ProgressUpdate::WriteProgress(file_complete_progress));
                                        
                                        if current_file == count {
                                            let _ = main_tx.send(ProgressUpdate::Done);
                                        } else {
                                            let _ = main_tx.send(ProgressUpdate::Log(
                                                format!("Completed file {} of {}", current_file, count)
                                            ));
                                        }
                                        break;
                                    }
                                }
                            }
                        });

                        // Use rounded_nonces for both the sequence spacing AND the file size
                        let task = HasherTask {
                            numeric_id,
                            start_nonce: current_start + (i * rounded_nonces),
                            nonces: rounded_nonces,
                            output_path: output_path_clone.clone(),
                            mem: memory_clone.clone(),
                            cpu_threads,
                            gpus: gpus_clone.clone(),
                            direct_io,
                            benchmark,
                            #[cfg(feature = "opencl")]
                            zcb: zero_copy,
                            #[cfg(feature = "gui")]
                            progress_tx: Some(file_tx),
                            stop_flag: Some(stop_flag_clone.clone()),
                        };
                        
                        let _ = tx_clone.send(ProgressUpdate::Log(
                            format!("Starting hashing {} of {} (nonce {})", current_file, count, current_start + (i * rounded_nonces))
                        ));
                        
                        hasher.run(task);
                        
                        // Wait for progress scaler to finish
                        let _ = progress_scaler.join();
                        
                        // Check stop flag after each file
                        if stop_flag_clone.load(Ordering::Relaxed) {
                            let _ = tx_clone.send(ProgressUpdate::Log(format!("Stop requested at file {} of {}", current_file, count)));
                            let _ = tx_clone.send(ProgressUpdate::Error("STOP_REQUESTED".to_string()));
                            break;
                        }
                    }
                });
            } else {
                // Single hashing mode - also use rounded_nonces
                let task = HasherTask {
                    numeric_id,
                    start_nonce,
                    nonces: rounded_nonces,
                    output_path,
                    mem: memory,
                    cpu_threads,
                    gpus,
                    direct_io,
                    benchmark,
                    #[cfg(feature = "opencl")]
                    zcb: zero_copy,
                    #[cfg(feature = "gui")]
                    progress_tx: Some(tx.clone()),
                    stop_flag: Some(stop_flag.clone()),
                };

                thread::spawn(move || {
                    hasher.run(task);
                    let _ = tx.send(ProgressUpdate::Done);
                });
            }
        }

    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    #[cfg(feature = "gui")]
    if args.len() == 1 {
        if let Err(e) = gui_app::launch() {
            eprintln!("Failed to start GUI: {}", e);
            process::exit(1);
        }
        return;
    }

    let mut cmd = Command::new("anne-hasher")
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about(env!("CARGO_PKG_DESCRIPTION"))
        .arg_required_else_help(true)
        .arg(
            Arg::new("gui")
                .short('g')
                .long("gui")
                .help("Launch graphical user interface")
                .action(ArgAction::SetTrue)
                .global(true),
        )
        .arg(
            Arg::new("disable_direct_io")
                .short('d')
                .long("ddio")
                .help("Disables direct i/o")
                .action(ArgAction::SetTrue)
                .global(true),
        )
        .arg(
            Arg::new("low_priority")
                .short('l')
                .long("prio")
                .help("Runs with low priority")
                .action(ArgAction::SetTrue)
                .global(true),
        )
        .arg(
            Arg::new("benchmark")
                .short('b')
                .long("bench")
                .help("Runs in xPU benchmark mode")
                .action(ArgAction::SetTrue)
                .global(true),
        )
        .arg(
            Arg::new("numeric_id")
                .short('i')
                .long("id")
                .value_name("NUMERIC_ID")
                .help("Your numeric Account ID")
                .value_parser(clap::value_parser!(u64))
                .required_unless_present("ocl_devices"),
        )
        .arg(
            Arg::new("start_nonce")
                .short('s')
                .long("sn")
                .value_name("START_NONCE")
                .help("Starting nonce for hashing")
                .value_parser(clap::value_parser!(u64))
                .required_unless_present("start_nonce_auto")
                .required_unless_present("ocl_devices"),
        )
        .arg(
            Arg::new("start_nonce_auto")
                .short('A')
                .long("sna")
                .value_name("COUNT")
                .help("Auto-hashing COUNT (>=1) sequential files, each with --n nonces, starting after the last existing hash segment for this ID. Ignores --sn.")
                .value_parser(clap::value_parser!(u64))
                .conflicts_with("start_nonce"),
        )
        .arg(
            Arg::new("nonces")
                .short('n')
                .long("n")
                .value_name("NONCES")
                .help("How many nonces you want to add")
                .value_parser(clap::value_parser!(u64))
                .required_unless_present("ocl_devices"),
        )
        .arg(
            Arg::new("path")
                .short('p')
                .long("path")
                .value_name("PATH")
                .help("Target path for hashfile (optional)"),
        )
        .arg(
            Arg::new("memory")
                .short('m')
                .long("mem")
                .value_name("MEMORY")
                .help("Maximum memory usage (optional)")
                .default_value("0B"),
        )
        .arg(
            Arg::new("cpu")
                .short('c')
                .long("cpu")
                .value_name("THREADS")
                .help("Maximum cpu cores you want to use (optional)")
                .value_parser(clap::value_parser!(u8)),
        )
        .arg(
            Arg::new("gpu")
                .short('g')
                .long("gpu")
                .value_name("platform_id:device_id:cores")
                .help("GPU(s) you want to use for hashing (optional)")
                .action(ArgAction::Append),
        )
        .group(
            ArgGroup::new("processing")
                .args(["cpu", "gpu"])
                .multiple(true),
        );

    #[cfg(feature = "opencl")]
    {
        cmd = cmd
            .arg(
                Arg::new("ocl_devices")
                    .short('o')
                    .long("opencl")
                    .help("Display OpenCL platforms and devices")
                    .action(ArgAction::SetTrue)
                    .global(true),
            )
            .arg(
                Arg::new("zero_copy")
                    .short('z')
                    .long("zcb")
                    .help("Enables zero copy buffers for shared mem (integrated) gpus")
                    .action(ArgAction::SetTrue)
                    .global(true),
            );
    }

    let matches = cmd.get_matches();

    if matches.get_flag("gui") {
        #[cfg(feature = "gui")]
        {
            if let Err(e) = gui_app::launch() {
                eprintln!("Failed to start GUI: {}", e);
                process::exit(1);
            }
            return;
        }
    }

    if matches.get_flag("low_priority") {
        set_low_prio();
    }

    #[cfg(feature = "opencl")]
    if matches.get_flag("ocl_devices") {
        ocl::platform_info();
        return;
    }

    let numeric_id = *matches.get_one::<u64>("numeric_id").expect("numeric_id required");

    let nonces = *matches.get_one::<u64>("nonces").expect("nonces required");

    let output_path = matches
        .get_one::<String>("path")
        .cloned()
        .unwrap_or_else(|| {
            std::env::current_dir()
                .unwrap()
                .into_os_string()
                .into_string()
                .unwrap()
        });

    let mem = matches.get_one::<String>("memory").cloned().unwrap();

    let cpu_threads_input = matches.get_one::<u8>("cpu").copied().unwrap_or(0);

    let gpus: Option<Vec<String>> = matches
        .get_many::<String>("gpu")
        .map(|v| v.cloned().collect());

    let cores = sys_info::cpu_num().unwrap() as u8;
    let mut cpu_threads = if cpu_threads_input == 0 {
        cores
    } else {
        min(2 * cores, cpu_threads_input)
    };

    #[cfg(feature = "opencl")]
    if matches.contains_id("gpu") && !matches.contains_id("cpu") {
        cpu_threads = 0;
    }

    let p = Hasher::new();

    if let Some(&auto_count) = matches.get_one::<u64>("start_nonce_auto") {
        if auto_count == 0 {
            eprintln!("Error: --sna count must be >= 1");
            process::exit(1);
        }

        println!("--sna enabled: hashing {auto_count} sequential file(s)");

        let mut current_start = 0u64;

        let rounded_nonces = if !matches.get_flag("disable_direct_io") {
            calculate_rounded_nonces(nonces, true, &output_path)
        } else {
            nonces
        };

        if let Ok(entries) = std::fs::read_dir(&output_path) {
            let mut max_end: u64 = 0;
            let prefix = format!("{}_", numeric_id);

            for entry in entries.flatten() {
                if let Some(file_name) = entry.file_name().to_str() {
                    if file_name.starts_with(&prefix) {
                        let parts: Vec<&str> = file_name.split('_').collect();
                        if parts.len() >= 3 {
                            if let (Ok(sn), Ok(cnt)) = (parts[1].parse::<u64>(), parts[2].parse::<u64>()) {
                                let end = sn + cnt;
                                if end > max_end {
                                    max_end = end;
                                }
                            }
                        }
                    }
                }
            }
            current_start = max_end;
        }

        println!("Starting from nonce {current_start}");
        if rounded_nonces != nonces {
            println!("Using rounded nonces per file: {} (original: {})", rounded_nonces, nonces);
        }

        for i in 0..auto_count {
            let this_start = current_start + i * rounded_nonces;

            println!("\n--- Hashing file {} of {auto_count}: start_nonce = {this_start} ---", i + 1);

            let file_task = HasherTask {
                numeric_id,
                start_nonce: this_start,
                nonces: rounded_nonces,
                output_path: output_path.clone(),
                mem: mem.clone(),
                cpu_threads,
                gpus: gpus.clone(),
                direct_io: !matches.get_flag("disable_direct_io"),
                benchmark: matches.get_flag("benchmark"),
                #[cfg(feature = "opencl")]
                zcb: matches.get_flag("zero_copy"),
                #[cfg(feature = "gui")]
                progress_tx: None,
                stop_flag: None,
            };

            p.run(file_task);
        }
    } else {
        let start_nonce = *matches.get_one::<u64>("start_nonce").expect("--sn is required when not using --sna");
        

        let final_nonces = if !matches.get_flag("disable_direct_io") {
            calculate_rounded_nonces(nonces, true, &output_path)
        } else {
            nonces
        };

        if final_nonces != nonces {
            println!("Using rounded nonces: {} (original: {})", final_nonces, nonces);
        }

        p.run(HasherTask {
            numeric_id,
            start_nonce,
            nonces: final_nonces,
            output_path,
            mem,
            cpu_threads,
            gpus,
            direct_io: !matches.get_flag("disable_direct_io"),
            benchmark: matches.get_flag("benchmark"),
            #[cfg(feature = "opencl")]
            zcb: matches.get_flag("zero_copy"),
            #[cfg(feature = "gui")]
            progress_tx: None,
            stop_flag: None,
        });
    }
}