const pkg = import("../pkg");
import Widget from "./Widget";

export const ui = Widget;
export const processor = pkg;
export const name = () => "birds";
export const description = () => "Plugin to classify birds on images.";
export const process = async (config, image) => {
  return import("../pkg").then((processor) => processor.process(config, image));
};
export const defaultConfig = async () => {
  return { x: 0, y: 0, size: 12.5, print: true, exif: false };
};
