//! Rust port of `ij.process` — ImageJ's core image-processing package.
//!
//! This crate is the starting point of a long-term, 1:1 port of ImageJ
//! (https://github.com/imagej/ImageJ, public domain) from Java to Rust.
//!
//! Mapping from the original Java packages:
//! - `ij.process.Blitter`  -> [`blitter`]
//!
//! Porting philosophy:
//! - Start with dependency-free, pure logic (blend modes, thresholds, stats).
//! - Keep each translated unit testable against the Java reference behavior.
//! - No PII / no personal data in any generated file.

pub mod blitter;
pub mod stats;
pub mod thresh;
