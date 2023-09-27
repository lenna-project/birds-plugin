use image::DynamicImage;
use std::io::Cursor;
use tract_onnx::prelude::*;

use crate::{Birds, ModelType};

const SIZE: usize = 226;

impl Birds {
    pub fn model() -> Result<ModelType, Box<dyn std::error::Error>> {
        let data = include_bytes!("../assets/birds_mobilenetv2.onnx");
        let mut cursor = Cursor::new(data);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, SIZE, SIZE, 3)),
            )?
            .with_output_fact(0, Default::default())?
            .into_optimized()?
            .into_runnable()?;
        Ok(model)
    }

    pub fn labels() -> Vec<String> {
        let collect = include_str!("../assets/birds_labels.txt")
            .to_string()
            .lines()
            .map(|s| s.to_string())
            .collect();
        collect
    }

    pub fn detect_label(
        &self,
        image: &Box<DynamicImage>,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let image_rgb = image.to_rgb8();
        let resized = image::imageops::resize(
            &image_rgb,
            SIZE as u32,
            SIZE as u32,
            ::image::imageops::FilterType::Triangle,
        );
        let tensor: Tensor =
            tract_ndarray::Array4::from_shape_fn((1, SIZE, SIZE, 3), |(_, y, x, c)| {
                let mean = [0.485, 0.456, 0.406][c];
                let std = [0.229, 0.224, 0.225][c];
                (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
            })
            .into();

        let result = self.model.run(tvec!(tensor.into())).unwrap();
        let best = result[0]
            .to_array_view::<f32>()?
            .iter()
            .cloned()
            .zip(0..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let index = best.unwrap().1;
        let label = Self::labels()[index].to_string();
        Ok(Some(label))
    }
}
