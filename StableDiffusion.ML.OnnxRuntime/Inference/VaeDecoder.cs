using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using StableDiffusion.ML.OnnxRuntime.Helpers;

namespace StableDiffusion.ML.OnnxRuntime.Inference
{
    public class VaeDecoder : IDisposable
    {
        private readonly SessionOptions _sessionOptions;
        private readonly InferenceSession _inferenceSession;
        private readonly StableDiffusionConfig _configuration;

        /// <summary>
        /// Initializes a new instance of the <see cref="VaeDecoder"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public VaeDecoder(StableDiffusionConfig configuration, PrePackedWeightsContainer prePackedWeightsContainer)
        {
            _configuration = configuration;
            _sessionOptions = _configuration.GetSessionOptionsForEp();
            _inferenceSession = new InferenceSession(_configuration.VaeDecoderOnnxPath, _sessionOptions, prePackedWeightsContainer);
        }


        /// <summary>
        /// Decodes the specified input.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <returns></returns>
        public Tensor<float> Decode(List<NamedOnnxValue> input)
        {
            using (var output = _inferenceSession.Run(input))
            {
                var result = output.FirstElementAs<Tensor<float>>();
                return result.Clone();
            }
        }


        /// <summary>
        /// Converts Tensor to Image.
        /// </summary>
        /// <param name="output">The output.</param>
        /// <param name="config">The configuration.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns></returns>
        public Image<Rgba32> ConvertToImage(Tensor<float> output)
        {
            var result = new Image<Rgba32>(_configuration.Width, _configuration.Height);

            for (var y = 0; y < _configuration.Height; y++)
            {
                for (var x = 0; x < _configuration.Width; x++)
                {
                    result[x, y] = new Rgba32(
                        (byte)Math.Round(Math.Clamp(output[0, 0, y, x] / 2 + 0.5, 0, 1) * 255),
                        (byte)Math.Round(Math.Clamp(output[0, 1, y, x] / 2 + 0.5, 0, 1) * 255),
                        (byte)Math.Round(Math.Clamp(output[0, 2, y, x] / 2 + 0.5, 0, 1) * 255)
                    );
                }
            }

            var imageName = $"sd_image_{DateTime.Now.ToString("yyyyMMddHHmm")}.png";
            var imagePath = Path.Combine(Directory.GetCurrentDirectory(), _configuration.ImageOutputPath, imageName);

            result.Save(imagePath);

            Console.WriteLine($"Image saved to: {imagePath}");

            return result;
        }

        public void Dispose()
        {
            _inferenceSession?.Dispose();
            _sessionOptions?.Dispose();
        }
    }
}
