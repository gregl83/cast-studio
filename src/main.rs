use opencv::{
    core::{Mat, Point, Scalar, Size, CV_8UC3, AlgorithmHint},
    highgui::{self, WINDOW_AUTOSIZE},
    imgcodecs,
    imgproc::{self, COLOR_BGR2HSV, FONT_HERSHEY_SIMPLEX},
    prelude::*,
    videoio::{self, VideoCapture, CAP_ANY},
};
use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use serde::{Deserialize, Serialize};
use tokio::time::interval;

// Configuration structure
#[derive(Debug, Serialize, Deserialize)]
struct Config {
    webcam_index: i32,
    output_width: i32,
    output_height: i32,
    background_type: BackgroundType,
    background_path: String,
    green_screen: GreenScreenConfig,
    overlays: Vec<OverlayConfig>,
    virtual_camera: VirtualCameraConfig,
}

#[derive(Debug, Serialize, Deserialize)]
enum BackgroundType {
    Image,
    Video,
}

#[derive(Debug, Serialize, Deserialize)]
struct GreenScreenConfig {
    hue_min: i32,
    hue_max: i32,
    saturation_min: i32,
    saturation_max: i32,
    value_min: i32,
    value_max: i32,
    blur_kernel_size: i32,
    morphology_kernel_size: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OverlayConfig {
    id: String,
    overlay_type: OverlayType,
    position: Position,
    size: Size2D,
    content: String,
    update_interval_ms: u64,
    font_scale: f64,
    color: [i32; 3],
    thickness: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
enum OverlayType {
    Text,
    Image,
    WebData,
    Logo,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Position {
    x: i32,
    y: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Size2D {
    width: i32,
    height: i32,
}

#[derive(Debug, Serialize, Deserialize)]
struct VirtualCameraConfig {
    device_name: String,
    fps: f64,
    format: String,
}

// Overlay data structure
#[derive(Debug, Clone)]
struct OverlayData {
    content: String,
    image: Option<Mat>,
    last_update: Instant,
}

// Main application structure
struct GreenScreenApp {
    config: Config,
    webcam: VideoCapture,
    background_image: Option<Mat>,
    background_video: Option<VideoCapture>,
    current_background_frame: Mat,
    overlay_data: Arc<Mutex<HashMap<String, OverlayData>>>,
    frame_count: u64,
    virtual_camera_writer: Option<opencv::videoio::VideoWriter>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            webcam_index: 0,
            output_width: 1280,
            output_height: 720,
            background_type: BackgroundType::Image,
            background_path: "background.jpg".to_string(),
            green_screen: GreenScreenConfig {
                hue_min: 35,
                hue_max: 85,
                saturation_min: 50,
                saturation_max: 255,
                value_min: 50,
                value_max: 255,
                blur_kernel_size: 5,
                morphology_kernel_size: 3,
            },
            overlays: vec![
                OverlayConfig {
                    id: "time".to_string(),
                    overlay_type: OverlayType::Text,
                    position: Position { x: 50, y: 50 },
                    size: Size2D { width: 200, height: 50 },
                    content: "Current Time".to_string(),
                    update_interval_ms: 1000,
                    font_scale: 1.0,
                    color: [255, 255, 255],
                    thickness: 2,
                },
            ],
            virtual_camera: VirtualCameraConfig {
                device_name: "/dev/video2".to_string(),
                fps: 30.0,
                format: "MJPG".to_string(),
            },
        }
    }
}

impl GreenScreenApp {
    fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize webcam
        let mut webcam = VideoCapture::new(config.webcam_index, CAP_ANY)?;
        webcam.set(videoio::CAP_PROP_FRAME_WIDTH, config.output_width as f64)?;
        webcam.set(videoio::CAP_PROP_FRAME_HEIGHT, config.output_height as f64)?;
        webcam.set(videoio::CAP_PROP_FPS, config.virtual_camera.fps)?;

        // Load background
        let (background_image, background_video) = match config.background_type {
            BackgroundType::Image => {
                let img = imgcodecs::imread(&config.background_path, imgcodecs::IMREAD_COLOR)?;
                let mut resized = Mat::default();
                imgproc::resize(
                    &img,
                    &mut resized,
                    Size::new(config.output_width, config.output_height),
                    0.0,
                    0.0,
                    imgproc::INTER_LINEAR,
                )?;
                (Some(resized), None)
            }
            BackgroundType::Video => {
                let video = VideoCapture::from_file(&config.background_path, CAP_ANY)?;
                (None, Some(video))
            }
        };

        let current_background_frame = Mat::zeros(
            config.output_height,
            config.output_width,
            CV_8UC3,
        )?
            .to_mat()?;

        // Initialize virtual camera writer
        let fourcc = opencv::videoio::VideoWriter::fourcc(
            'M', 'J', 'P', 'G'
        )?;

        let virtual_camera_writer = opencv::videoio::VideoWriter::new(
            &config.virtual_camera.device_name,
            fourcc,
            config.virtual_camera.fps,
            Size::new(config.output_width, config.output_height),
            true,
        ).ok();

        Ok(GreenScreenApp {
            config,
            webcam,
            background_image,
            background_video,
            current_background_frame,
            overlay_data: Arc::new(Mutex::new(HashMap::new())),
            frame_count: 0,
            virtual_camera_writer,
        })
    }

    fn remove_green_screen(&self, frame: &Mat) -> Result<Mat, Box<dyn std::error::Error>> {
        // Convert to HSV color space
        let mut hsv = Mat::default();
        imgproc::cvt_color(frame, &mut hsv, COLOR_BGR2HSV, 0, AlgorithmHint::default())?;

        // Create mask for green screen
        let lower_green = Scalar::new(
            self.config.green_screen.hue_min as f64,
            self.config.green_screen.saturation_min as f64,
            self.config.green_screen.value_min as f64,
            0.0,
        );
        let upper_green = Scalar::new(
            self.config.green_screen.hue_max as f64,
            self.config.green_screen.saturation_max as f64,
            self.config.green_screen.value_max as f64,
            255.0,
        );

        let mut mask = Mat::default();
        opencv::core::in_range(&hsv, &lower_green, &upper_green, &mut mask)?;

        // Apply morphological operations to clean up the mask
        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_ELLIPSE,
            Size::new(
                self.config.green_screen.morphology_kernel_size,
                self.config.green_screen.morphology_kernel_size,
            ),
            Point::new(-1, -1),
        )?;

        let mut cleaned_mask = Mat::default();
        imgproc::morphology_ex(
            &mask,
            &mut cleaned_mask,
            imgproc::MORPH_CLOSE,
            &kernel,
            Point::new(-1, -1),
            1,
            opencv::core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?,
        )?;

        // Apply Gaussian blur to soften edges
        let mut blurred_mask = Mat::default();
        imgproc::gaussian_blur(
            &cleaned_mask,
            &mut blurred_mask,
            Size::new(
                self.config.green_screen.blur_kernel_size,
                self.config.green_screen.blur_kernel_size,
            ),
            0.0,
            0.0,
            opencv::core::BORDER_DEFAULT,
            AlgorithmHint::default(),
        )?;

        Ok(blurred_mask)
    }

    fn update_background_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match &self.config.background_type {
            BackgroundType::Image => {
                if let Some(ref bg_img) = self.background_image {
                    bg_img.copy_to(&mut self.current_background_frame)?;
                }
            }
            BackgroundType::Video => {
                if let Some(ref mut bg_video) = self.background_video {
                    let mut frame = Mat::default();
                    if bg_video.read(&mut frame)? && !frame.empty() {
                        let mut resized = Mat::default();
                        imgproc::resize(
                            &frame,
                            &mut resized,
                            Size::new(self.config.output_width, self.config.output_height),
                            0.0,
                            0.0,
                            imgproc::INTER_LINEAR,
                        )?;
                        resized.copy_to(&mut self.current_background_frame)?;
                    } else {
                        // Loop video - restart from beginning
                        bg_video.set(videoio::CAP_PROP_POS_FRAMES, 0.0)?;
                        if bg_video.read(&mut frame)? && !frame.empty() {
                            let mut resized = Mat::default();
                            imgproc::resize(
                                &frame,
                                &mut resized,
                                Size::new(self.config.output_width, self.config.output_height),
                                0.0,
                                0.0,
                                imgproc::INTER_LINEAR,
                            )?;
                            resized.copy_to(&mut self.current_background_frame)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn apply_overlays(&self, frame: &mut Mat) -> Result<(), Box<dyn std::error::Error>> {
        let overlay_data = self.overlay_data.lock().unwrap();

        for overlay_config in &self.config.overlays {
            if let Some(data) = overlay_data.get(&overlay_config.id) {
                match overlay_config.overlay_type {
                    OverlayType::Text => {
                        let color = Scalar::new(
                            overlay_config.color[0] as f64,
                            overlay_config.color[1] as f64,
                            overlay_config.color[2] as f64,
                            255.0,
                        );

                        imgproc::put_text(
                            frame,
                            &data.content,
                            Point::new(overlay_config.position.x, overlay_config.position.y),
                            FONT_HERSHEY_SIMPLEX,
                            overlay_config.font_scale,
                            color,
                            overlay_config.thickness,
                            imgproc::LINE_8,
                            false,
                        )?;
                    }
                    OverlayType::Image | OverlayType::Logo => {
                        if let Some(ref overlay_img) = data.image {
                            // Create ROI for overlay
                            let roi = opencv::core::Rect::new(
                                overlay_config.position.x,
                                overlay_config.position.y,
                                overlay_config.size.width.min(frame.cols() - overlay_config.position.x),
                                overlay_config.size.height.min(frame.rows() - overlay_config.position.y),
                            );

                            if roi.width > 0 && roi.height > 0 {
                                let mut roi_frame = Mat::roi(frame, roi)?;
                                let mut resized_overlay = Mat::default();
                                imgproc::resize(
                                    overlay_img,
                                    &mut resized_overlay,
                                    Size::new(roi.width, roi.height),
                                    0.0,
                                    0.0,
                                    imgproc::INTER_LINEAR,
                                )?;
                                resized_overlay.copy_to(&mut roi_frame.try_deref_mut()?)?;
                            }
                        }
                    }
                    OverlayType::WebData => {
                        // Similar to text overlay but with web-fetched data
                        let color = Scalar::new(
                            overlay_config.color[0] as f64,
                            overlay_config.color[1] as f64,
                            overlay_config.color[2] as f64,
                            255.0,
                        );

                        imgproc::put_text(
                            frame,
                            &data.content,
                            Point::new(overlay_config.position.x, overlay_config.position.y),
                            FONT_HERSHEY_SIMPLEX,
                            overlay_config.font_scale,
                            color,
                            overlay_config.thickness,
                            imgproc::LINE_8,
                            false,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn composite_frame(&self, webcam_frame: &Mat, mask: &Mat) -> Result<Mat, Box<dyn std::error::Error>> {
        let mut result = Mat::default();

        // Convert single channel mask to 3-channel
        let mut mask_3ch = Mat::default();
        imgproc::cvt_color(mask, &mut mask_3ch, imgproc::COLOR_GRAY2BGR, 0, AlgorithmHint::default())?;

        // Normalize mask to 0-1 range
        let mut mask_normalized = Mat::default();
        mask_3ch.convert_to(&mut mask_normalized, opencv::core::CV_32F, 1.0/255.0, 0.0)?;

        // Convert frames to float for blending
        let mut webcam_float = Mat::default();
        let mut background_float = Mat::default();
        webcam_frame.convert_to(&mut webcam_float, opencv::core::CV_32F, 1.0, 0.0)?;
        self.current_background_frame.convert_to(&mut background_float, opencv::core::CV_32F, 1.0, 0.0)?;

        // Create inverse mask
        let mut inv_mask = Mat::default();
        opencv::core::subtract(&Scalar::all(1.0), &mask_normalized, &mut inv_mask, &opencv::core::no_array(), -1)?;

        // Blend: result = webcam * (1 - mask) + background * mask
        let mut webcam_masked = Mat::default();
        let mut background_masked = Mat::default();
        opencv::core::multiply(&webcam_float, &inv_mask, &mut webcam_masked, 1.0, -1)?;
        opencv::core::multiply(&background_float, &mask_normalized, &mut background_masked, 1.0, -1)?;

        let mut result_float = Mat::default();
        opencv::core::add(&webcam_masked, &background_masked, &mut result_float, &opencv::core::no_array(), -1)?;

        // Convert back to 8-bit
        result_float.convert_to(&mut result, opencv::core::CV_8U, 1.0, 0.0)?;

        Ok(result)
    }

    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Start overlay update threads
        self.start_overlay_threads()?;

        println!("Starting green screen application...");
        println!("Press 'q' to quit, 's' to save current frame");

        let window_name = "Green Screen Virtual Webcam";
        highgui::named_window(window_name, WINDOW_AUTOSIZE)?;

        loop {
            // Capture webcam frame
            let mut webcam_frame = Mat::default();
            if !self.webcam.read(&mut webcam_frame)? || webcam_frame.empty() {
                println!("Failed to capture frame from webcam");
                break;
            }

            // Update background frame
            self.update_background_frame()?;

            // Remove green screen
            let mask = self.remove_green_screen(&webcam_frame)?;

            // Composite with background
            let mut result = self.composite_frame(&webcam_frame, &mask)?;

            // Apply overlays
            self.apply_overlays(&mut result)?;

            // Write to virtual camera
            if let Some(ref mut writer) = self.virtual_camera_writer {
                writer.write(&result)?;
            }

            // Display result
            highgui::imshow(window_name, &result)?;

            // Handle key presses
            let key = highgui::wait_key(1)?;
            if key == 113 { // 'q' key
                break;
            } else if key == 115 { // 's' key
                let filename = format!("frame_{}.jpg", self.frame_count);
                imgcodecs::imwrite(&filename, &result, &opencv::core::Vector::new())?;
                println!("Saved frame: {}", filename);
            }

            self.frame_count += 1;
        }

        highgui::destroy_all_windows()?;
        Ok(())
    }

    fn start_overlay_threads(&self) -> Result<(), Box<dyn std::error::Error>> {
        let overlay_data = Arc::clone(&self.overlay_data);
        let overlays = self.config.overlays.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100));

            loop {
                interval.tick().await;

                for overlay_config in &overlays {
                    // Check if it's time to update this overlay
                    let should_update = {
                        let data = overlay_data.lock().unwrap();
                        match data.get(&overlay_config.id) {
                            Some(overlay) => {
                                overlay.last_update.elapsed().as_millis() >= overlay_config.update_interval_ms as u128
                            }
                            None => true,
                        }
                    };

                    if should_update {
                        let content = match overlay_config.overlay_type {
                            OverlayType::Text => {
                                match overlay_config.content.as_str() {
                                    "Current Time" => {
                                        chrono::Local::now().format("%H:%M:%S").to_string()
                                    }
                                    _ => overlay_config.content.clone(),
                                }
                            }
                            OverlayType::WebData => {
                                // Fetch data from web API
                                match reqwest::get(&overlay_config.content).await {
                                    Ok(response) => {
                                        response.text().await.unwrap_or_else(|_| "Error".to_string())
                                    }
                                    Err(_) => "Network Error".to_string(),
                                }
                            }
                            OverlayType::Image | OverlayType::Logo => {
                                // Load image
                                overlay_config.content.clone()
                            }
                        };

                        let image = match overlay_config.overlay_type {
                            OverlayType::Image | OverlayType::Logo => {
                                imgcodecs::imread(&overlay_config.content, imgcodecs::IMREAD_COLOR).ok()
                            }
                            _ => None,
                        };

                        let mut data = overlay_data.lock().unwrap();
                        data.insert(
                            overlay_config.id.clone(),
                            OverlayData {
                                content,
                                image,
                                last_update: Instant::now(),
                            },
                        );
                    }
                }
            }
        });

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load or create configuration
    let config_path = "config.json";
    let config = if Path::new(config_path).exists() {
        let config_str = fs::read_to_string(config_path)?;
        serde_json::from_str(&config_str)?
    } else {
        let default_config = Config::default();
        let config_str = serde_json::to_string_pretty(&default_config)?;
        fs::write(config_path, config_str)?;
        println!("Created default config file: {}", config_path);
        default_config
    };

    // Create and run the application
    let mut app = GreenScreenApp::new(config)?;
    app.run()?;

    Ok(())
}

// Additional utility functions for v4l2loopback setup
#[cfg(target_os = "linux")]
mod v4l2_utils {
    use std::process::Command;

    pub fn setup_virtual_camera() -> Result<(), Box<dyn std::error::Error>> {
        // Load v4l2loopback module
        let output = Command::new("sudo")
            .args(&["modprobe", "v4l2loopback", "devices=1", "video_nr=2", "card_label=GreenScreenCam"])
            .output()?;

        if !output.status.success() {
            eprintln!("Failed to load v4l2loopback module. Make sure it's installed:");
            eprintln!("sudo apt install v4l2loopback-dkms");
            return Err("v4l2loopback setup failed".into());
        }

        println!("Virtual camera device created at /dev/video2");
        Ok(())
    }

    pub fn cleanup_virtual_camera() -> Result<(), Box<dyn std::error::Error>> {
        let _output = Command::new("sudo")
            .args(&["modprobe", "-r", "v4l2loopback"])
            .output()?;

        println!("Virtual camera cleanup complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.webcam_index, deserialized.webcam_index);
        assert_eq!(config.output_width, deserialized.output_width);
    }

    #[test]
    fn test_overlay_config() {
        let overlay = OverlayConfig {
            id: "test".to_string(),
            overlay_type: OverlayType::Text,
            position: Position { x: 10, y: 20 },
            size: Size2D { width: 100, height: 50 },
            content: "Test Text".to_string(),
            update_interval_ms: 1000,
            font_scale: 1.0,
            color: [255, 255, 255],
            thickness: 2,
        };

        assert_eq!(overlay.id, "test");
        assert_eq!(overlay.position.x, 10);
    }
}