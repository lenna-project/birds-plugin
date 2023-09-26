use image::DynamicImage;
use std::io::Cursor;
use tract_onnx::prelude::*;

use crate::{Birds, ModelType};

const SIZE: usize = 260;

impl Birds {
    pub fn model() -> Result<ModelType, Box<dyn std::error::Error>> {
        let data = include_bytes!("../assets/birds_efficientnetb2.onnx");
        let mut cursor = Cursor::new(data);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, SIZE, SIZE)),
            )?
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
            tract_ndarray::Array4::from_shape_fn((1, 3, SIZE, SIZE), |(_, c, y, x)| {
                (resized[(x as _, y as _)][c] as f32 / 255.0)
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
