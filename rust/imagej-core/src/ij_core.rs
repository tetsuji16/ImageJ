//! Core ImageJ data structures: ImagePlus, ImageStack, Calibration, FileInfo.
//!
//! Mirrors `ij.ImagePlus`, `ij.ImageStack`, `ij.Calibration`, `ij.io.FileInfo`.
//! GUI/AWT-dependent fields (ImageWindow, ImageCanvas, ROI, etc.) are omitted
//! or represented as optional/placeholder types.

use crate::io::ImageData;
use crate::processor::ByteProcessor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Image type constants. Mirrors `ImagePlus.GRAY8`, `GRAY16`, `GRAY32`, `COLOR_256`, `COLOR_RGB`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i32)]
pub enum ImageType {
    Gray8 = 0,
    Gray16 = 1,
    Gray32 = 2,
    Color256 = 3,
    ColorRgb = 4,
}

impl ImageType {
    pub fn from_processor(data: &ImageData) -> Self {
        match data {
            ImageData::Byte(_) => ImageType::Gray8,
            ImageData::Short(_) => ImageType::Gray16,
            ImageData::Float(_) => ImageType::Gray32,
            ImageData::Color(_) => ImageType::ColorRgb,
        }
    }
}

/// Calibration information (spatial + density). Mirrors `ij.Calibration`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    /// Pixel width in calibrated units (e.g., µm)
    pub pixel_width: f64,
    /// Pixel height in calibrated units
    pub pixel_height: f64,
    /// Pixel depth (voxel depth) in calibrated units
    pub pixel_depth: f64,
    /// Unit of measurement (e.g., "µm", "mm", "cm", "pixels")
    pub unit: String,
    /// Minimum displayed pixel value
    pub display_min: f64,
    /// Maximum displayed pixel value
    pub display_max: f64,
    /// Calibration function (e.g., "linear", "log", "custom")
    pub function: String,
    /// Coefficients for calibration function
    pub coefficients: Vec<f64>,
    /// Origin (x, y, z) in calibrated units
    pub x_origin: f64,
    pub y_origin: f64,
    pub z_origin: f64,
    /// Frame interval (for time-lapse)
    pub frame_interval: f64,
    /// Time unit
    pub time_unit: String,
    /// Global calibration flag
    pub global: bool,
}

impl Default for Calibration {
    fn default() -> Self {
        Calibration {
            pixel_width: 1.0,
            pixel_height: 1.0,
            pixel_depth: 1.0,
            unit: "pixels".to_string(),
            display_min: 0.0,
            display_max: 255.0,
            function: "linear".to_string(),
            coefficients: vec![],
            x_origin: 0.0,
            y_origin: 0.0,
            z_origin: 0.0,
            frame_interval: 1.0,
            time_unit: "sec".to_string(),
            global: false,
        }
    }
}

/// File metadata. Mirrors `ij.io.FileInfo`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub file_name: String,
    pub directory: String,
    pub url: Option<String>,
    pub width: usize,
    pub height: usize,
    pub n_images: usize,
    pub offset: usize,
    pub gap: usize,
    pub white_is_zero: bool,
    pub intellectual_property: Option<String>,
    pub file_type: String, // e.g., "TIFF", "PNG", "JPEG"
    pub bit_depth: usize,
    pub compression: String,
    pub n_channels: usize,
    pub n_slices: usize,
    pub n_frames: usize,
    pub interval: f64,
    pub metadata: HashMap<String, String>,
}

impl Default for FileInfo {
    fn default() -> Self {
        FileInfo {
            file_name: String::new(),
            directory: String::new(),
            url: None,
            width: 0,
            height: 0,
            n_images: 1,
            offset: 0,
            gap: 0,
            white_is_zero: false,
            intellectual_property: None,
            file_type: String::new(),
            bit_depth: 8,
            compression: "none".to_string(),
            n_channels: 1,
            n_slices: 1,
            n_frames: 1,
            interval: 1.0,
            metadata: HashMap::new(),
        }
    }
}

/// A single slice label + pixels (any processor type).
/// Mirrors one entry in `ImageStack`.
#[derive(Debug, Clone)]
pub struct StackSlice {
    pub label: Option<String>,
    pub data: ImageData,
}

/// Image stack — ordered list of slices with same dimensions and type.
/// Mirrors `ij.ImageStack` (minus AWT ColorModel, ROI, viewers).
#[derive(Debug, Clone)]
pub struct ImageStack {
    pub width: usize,
    pub height: usize,
    pub slices: Vec<StackSlice>,
    pub bit_depth: usize, // 8, 16, 24, 32
    pub min: f64,
    pub max: f64,
}

impl ImageStack {
    /// Creates an empty stack with given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        ImageStack {
            width,
            height,
            slices: Vec::new(),
            bit_depth: 0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Returns number of slices.
    pub fn size(&self) -> usize {
        self.slices.len()
    }

    /// Adds a slice at the end.
    pub fn add_slice(&mut self, label: Option<String>, data: ImageData) {
        // Validate dimensions match
        let (w, h) = match &data {
            ImageData::Byte(bp) => (bp.width, bp.height),
            ImageData::Short(sp) => (sp.width, sp.height),
            ImageData::Float(fp) => (fp.width, fp.height),
            ImageData::Color(cp) => (cp.width, cp.height),
        };
        assert_eq!(w, self.width, "slice width mismatch");
        assert_eq!(h, self.height, "slice height mismatch");

        // Update bit depth if first slice
        let depth = match &data {
            ImageData::Byte(_) => 8,
            ImageData::Short(_) => 16,
            ImageData::Float(_) => 32,
            ImageData::Color(_) => 24,
        };
        if self.bit_depth == 0 {
            self.bit_depth = depth;
        } else {
            assert_eq!(self.bit_depth, depth, "bit depth mismatch");
        }

        // Update min/max
        let hist = match &data {
            ImageData::Byte(bp) => bp.get_histogram(),
            ImageData::Short(sp) => sp.get_histogram(),
            ImageData::Float(fp) => fp.get_histogram(),
            ImageData::Color(_) => [0u32; 65536],
        };
        for (i, &count) in hist.iter().enumerate() {
            if count > 0 {
                let val = i as f64;
                if val < self.min {
                    self.min = val;
                }
                if val > self.max {
                    self.max = val;
                }
            }
        }

        self.slices.push(StackSlice { label, data });
    }

    /// Gets a slice by index (0-based).
    pub fn get_slice(&self, index: usize) -> Option<&StackSlice> {
        self.slices.get(index)
    }

    /// Gets a mutable slice by index.
    pub fn get_slice_mut(&mut self, index: usize) -> Option<&mut StackSlice> {
        self.slices.get_mut(index)
    }

    /// Removes a slice by index.
    pub fn remove_slice(&mut self, index: usize) {
        self.slices.remove(index);
    }

    /// Converts stack to a vector of ByteProcessors (for 8-bit stacks).
    pub fn to_byte_processors(&self) -> Vec<ByteProcessor> {
        self.slices
            .iter()
            .map(|s| match &s.data {
                ImageData::Byte(bp) => bp.clone(),
                _ => panic!("stack not 8-bit"),
            })
            .collect()
    }
}

/// Core image container. Mirrors `ij.ImagePlus` (minus GUI/window/threading fields).
#[derive(Debug, Clone)]
pub struct ImagePlus {
    /// Unique ID (decreasing from 0, like Java's `currentID`).
    pub id: i32,
    /// Window title / image name.
    pub title: String,
    /// Image type (8-bit, 16-bit, 32-bit float, RGB).
    pub image_type: ImageType,
    /// Single 2D image processor (if not a stack).
    pub processor: Option<ImageData>,
    /// Image stack (3D/4D/5D).
    pub stack: Option<ImageStack>,
    /// Current slice index (1-based, like Java).
    pub current_slice: usize,
    /// Calibration info.
    pub calibration: Calibration,
    /// File metadata.
    pub file_info: Option<FileInfo>,
    /// Dimensions: channels, slices, frames.
    pub n_channels: usize,
    pub n_slices: usize,
    pub n_frames: usize,
    /// Custom properties (key-value).
    pub properties: HashMap<String, String>,
    /// Has unsaved changes.
    pub changes: bool,
    /// Overlay (shapes, ROIs) - placeholder.
    pub overlay: Option<Overlay>,
    /// ROI - placeholder (we'll add Roi later).
    pub roi: Option<Box<dyn Roi>>,
}

static mut CURRENT_ID: i32 = -1;

fn next_id() -> i32 {
    unsafe {
        CURRENT_ID -= 1;
        CURRENT_ID
    }
}

impl ImagePlus {
    /// Creates an uninitialized ImagePlus.
    pub fn new() -> Self {
        ImagePlus {
            id: next_id(),
            title: "null".to_string(),
            image_type: ImageType::Gray8,
            processor: None,
            stack: None,
            current_slice: 1,
            calibration: Calibration::default(),
            file_info: None,
            n_channels: 1,
            n_slices: 1,
            n_frames: 1,
            properties: HashMap::new(),
            changes: false,
            overlay: None,
            roi: None,
        }
    }

    /// Creates from an ImageData (single 2D image).
    pub fn from_image_data(title: &str, data: ImageData) -> Self {
        let mut imp = ImagePlus {
            id: next_id(),
            title: title.to_string(),
            image_type: ImageType::from_processor(&data),
            processor: Some(data),
            stack: None,
            current_slice: 1,
            calibration: Calibration::default(),
            file_info: None,
            n_channels: 1,
            n_slices: 1,
            n_frames: 1,
            properties: HashMap::new(),
            changes: false,
            overlay: None,
            roi: None,
        };
        imp.update_dimensions();
        imp
    }

    /// Creates from an ImageStack.
    pub fn from_stack(title: &str, stack: ImageStack) -> Self {
        let image_type = if stack.slices.is_empty() {
            ImageType::Gray8
        } else {
            ImageType::from_processor(&stack.slices[0].data)
        };
        let n_slices = stack.size();
        let mut imp = ImagePlus {
            id: next_id(),
            title: title.to_string(),
            image_type,
            processor: None,
            stack: Some(stack),
            current_slice: 1,
            calibration: Calibration::default(),
            file_info: None,
            n_channels: 1,
            n_slices,
            n_frames: 1,
            properties: HashMap::new(),
            changes: false,
            overlay: None,
            roi: None,
        };
        imp.update_dimensions();
        imp
    }

    /// Returns width in pixels.
    pub fn width(&self) -> usize {
        match &self.processor {
            Some(ImageData::Byte(bp)) => bp.width,
            Some(ImageData::Short(sp)) => sp.width,
            Some(ImageData::Float(fp)) => fp.width,
            Some(ImageData::Color(cp)) => cp.width,
            None => self.stack.as_ref().map(|s| s.width).unwrap_or(0),
        }
    }

    /// Returns height in pixels.
    pub fn height(&self) -> usize {
        match &self.processor {
            Some(ImageData::Byte(bp)) => bp.height,
            Some(ImageData::Short(sp)) => sp.height,
            Some(ImageData::Float(fp)) => fp.height,
            Some(ImageData::Color(cp)) => cp.height,
            None => self.stack.as_ref().map(|s| s.height).unwrap_or(0),
        }
    }

    /// Returns the processor for the current slice (or the single image).
    pub fn get_processor(&self) -> Option<&ImageData> {
        if let Some(stack) = &self.stack {
            stack.get_slice(self.current_slice.saturating_sub(1))
                .map(|s| &s.data)
        } else {
            self.processor.as_ref()
        }
    }

    /// Returns a mutable processor for the current slice.
    pub fn get_processor_mut(&mut self) -> Option<&mut ImageData> {
        if let Some(stack) = &mut self.stack {
            stack.get_slice_mut(self.current_slice.saturating_sub(1))
                .map(|s| &mut s.data)
        } else {
            self.processor.as_mut()
        }
    }

    /// Sets the processor for a single-image (non-stack) ImagePlus,
    /// updating the image type accordingly.
    pub fn set_processor(&mut self, ip: ImageData) {
        self.image_type = ImageType::from_processor(&ip);
        self.processor = Some(ip);
    }

    /// Sets the current slice (1-based).
    pub fn set_slice(&mut self, slice: usize) {
        let max = self.get_stack_size();
        if slice >= 1 && slice <= max {
            self.current_slice = slice;
        }
    }

    /// Returns total number of slices (for stacks).
    pub fn get_stack_size(&self) -> usize {
        self.stack.as_ref().map(|s| s.size()).unwrap_or(1)
    }

    /// Updates n_channels, n_slices, n_frames from stack/processor.
    fn update_dimensions(&mut self) {
        if let Some(stack) = &self.stack {
            self.n_slices = stack.size();
            // Heuristic: if stack size > 1, it's a z-stack or time series
        }
    }

    /// Sets calibration.
    pub fn set_calibration(&mut self, cal: Calibration) {
        self.calibration = cal;
    }

    /// Sets file info.
    pub fn set_file_info(&mut self, fi: FileInfo) {
        self.file_info = Some(fi);
    }

    /// Marks image as changed.
    pub fn set_changes(&mut self, changed: bool) {
        self.changes = changed;
    }
}

impl Default for ImagePlus {
    fn default() -> Self {
        Self::new()
    }
}

/// Placeholder for Overlay (shapes, ROIs drawn on top).
#[derive(Debug, Clone, Default)]
pub struct Overlay {
    pub items: Vec<OverlayItem>,
}

#[derive(Debug, Clone)]
pub enum OverlayItem {
    Roi(Box<dyn Roi>),
    Label { x: i32, y: i32, text: String, color: u32 },
}

/// Trait for ROI (Region of Interest). Mirrors `ij.gui.Roi` (subset).
pub trait Roi: std::fmt::Debug + Send + Sync {
    fn bounds(&self) -> (i32, i32, i32, i32); // x, y, width, height
    fn contains(&self, x: i32, y: i32) -> bool;
    fn get_type(&self) -> RoiType;
    fn clone_box(&self) -> Box<dyn Roi>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoiType {
    Rectangle,
    Oval,
    Polygon,
    Freehand,
    Line,
    Point,
    Composite,
}

/// Simple rectangular ROI.
#[derive(Debug, Clone)]
pub struct RectRoi {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Roi for RectRoi {
    fn bounds(&self) -> (i32, i32, i32, i32) {
        (self.x, self.y, self.width, self.height)
    }

    fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    fn get_type(&self) -> RoiType {
        RoiType::Rectangle
    }

    fn clone_box(&self) -> Box<dyn Roi> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Roi> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::{ByteProcessor, ColorProcessor, FloatProcessor, ShortProcessor};

    #[test]
    fn imageplus_from_byte_processor() {
        let bp = ByteProcessor::from_pixels(10, 10, vec![128u8; 100]);
        let imp = ImagePlus::from_image_data("test", ImageData::Byte(bp));
        assert_eq!(imp.title, "test");
        assert_eq!(imp.image_type, ImageType::Gray8);
        assert_eq!(imp.width(), 10);
        assert_eq!(imp.height(), 10);
        assert_eq!(imp.get_stack_size(), 1);
    }

    #[test]
    fn imageplus_from_short_processor() {
        let sp = ShortProcessor::from_pixels(5, 5, vec![100u16; 25]);
        let imp = ImagePlus::from_image_data("short_img", ImageData::Short(sp));
        assert_eq!(imp.image_type, ImageType::Gray16);
    }

    #[test]
    fn imageplus_from_float_processor() {
        let fp = FloatProcessor::from_pixels(4, 4, vec![1.5f32; 16]);
        let imp = ImagePlus::from_image_data("float_img", ImageData::Float(fp));
        assert_eq!(imp.image_type, ImageType::Gray32);
    }

    #[test]
    fn imageplus_from_color_processor() {
        let cp = ColorProcessor::from_pixels(3, 3, vec![0xFF00FF00u32; 9]);
        let imp = ImagePlus::from_image_data("color_img", ImageData::Color(cp));
        assert_eq!(imp.image_type, ImageType::ColorRgb);
    }

    #[test]
    fn imagestack_add_slices() {
        let mut stack = ImageStack::new(4, 4);
        let bp1 = ByteProcessor::from_pixels(4, 4, vec![1u8; 16]);
        let bp2 = ByteProcessor::from_pixels(4, 4, vec![2u8; 16]);
        stack.add_slice(Some("slice1".to_string()), ImageData::Byte(bp1));
        stack.add_slice(Some("slice2".to_string()), ImageData::Byte(bp2));
        assert_eq!(stack.size(), 2);
        assert_eq!(stack.get_slice(0).unwrap().label, Some("slice1".to_string()));
    }

    #[test]
    fn imageplus_from_stack() {
        let mut stack = ImageStack::new(2, 2);
        stack.add_slice(None, ImageData::Byte(ByteProcessor::from_pixels(2, 2, vec![1, 2, 3, 4])));
        stack.add_slice(None, ImageData::Byte(ByteProcessor::from_pixels(2, 2, vec![5, 6, 7, 8])));
        let mut imp = ImagePlus::from_stack("stack_img", stack);
        assert_eq!(imp.get_stack_size(), 2);
        assert_eq!(imp.current_slice, 1);
        imp.set_slice(2);
        assert_eq!(imp.current_slice, 2);
        let proc = imp.get_processor().unwrap();
        match proc {
            ImageData::Byte(bp) => assert_eq!(bp.pixels, vec![5, 6, 7, 8]),
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn calibration_default() {
        let cal = Calibration::default();
        assert_eq!(cal.pixel_width, 1.0);
        assert_eq!(cal.unit, "pixels");
    }

    #[test]
    fn fileinfo_default() {
        let fi = FileInfo::default();
        assert_eq!(fi.n_images, 1);
        assert_eq!(fi.bit_depth, 8);
    }

    #[test]
    fn rect_roi_bounds_contains() {
        let roi = RectRoi { x: 10, y: 20, width: 5, height: 5 };
        assert_eq!(roi.bounds(), (10, 20, 5, 5));
        assert!(roi.contains(12, 22));
        assert!(!roi.contains(9, 20));
        assert!(!roi.contains(15, 25));
    }
}