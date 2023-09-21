using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace StableDiffusion.ML.OnnxRuntime
{
    public class SafetyChecker : IDisposable
    {
        private readonly SessionOptions _sessionOptions;
        private readonly InferenceSession _inferenceSession;
        private readonly StableDiffusionConfig _configuration;

        /// <summary>
        /// Initializes a new instance of the <see cref="SafetyChecker"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public SafetyChecker(StableDiffusionConfig configuration)
        {
            _configuration = configuration;
            _sessionOptions = _configuration.GetSessionOptionsForEp();
            _inferenceSession = new InferenceSession(_configuration.SafetyModelPath, _sessionOptions);
        }


        /// <summary>
        /// Determines whether the specified result image is not NSFW.
        /// </summary>
        /// <param name="resultImage">The result image.</param>
        /// <param name="config">The configuration.</param>
        /// <returns>
        ///   <c>true</c> if the specified result image is safe; otherwise, <c>false</c>.
        /// </returns>
        public bool IsImageSafe(Tensor<float> resultImage, StableDiffusionConfig config)
        {
            //clip input
            var inputTensor = ClipImageFeatureExtractor(resultImage);

            //images input
            var inputImagesTensor = ReorderTensor(inputTensor);

            var input = new List<NamedOnnxValue>
            {
                //batch channel height width
                 NamedOnnxValue.CreateFromTensor("clip_input", inputTensor),

                 //batch, height, width, channel
                 NamedOnnxValue.CreateFromTensor("images", inputImagesTensor)
            };

            // Run session and send the input data in to get inference output. 
            using (var output = _inferenceSession.Run(input))
            {
                var result = output.LastElementAs<IEnumerable<bool>>();
                return !result.First();
            }
        }


        /// <summary>
        /// Reorders the tensor.
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <returns></returns>
        private DenseTensor<float> ReorderTensor(Tensor<float> inputTensor)
        {
            //reorder from batch channel height width to batch height width channel
            var inputImagesTensor = new DenseTensor<float>(new[] { 1, 224, 224, 3 });
            for (int y = 0; y < inputTensor.Dimensions[2]; y++)
            {
                for (int x = 0; x < inputTensor.Dimensions[3]; x++)
                {
                    inputImagesTensor[0, y, x, 0] = inputTensor[0, 0, y, x];
                    inputImagesTensor[0, y, x, 1] = inputTensor[0, 1, y, x];
                    inputImagesTensor[0, y, x, 2] = inputTensor[0, 2, y, x];
                }
            }

            return inputImagesTensor;
        }


        /// <summary>
        /// Image feature extractor.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns></returns>
        private DenseTensor<float> ClipImageFeatureExtractor(Tensor<float> imageTensor)
        {
            //convert tensor result to image
            var image = new Image<Rgba32>(_configuration.Width, _configuration.Height);

            for (var y = 0; y < _configuration.Height; y++)
            {
                for (var x = 0; x < _configuration.Width; x++)
                {
                    image[x, y] = new Rgba32(
                        (byte)(Math.Round(Math.Clamp((imageTensor[0, 0, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((imageTensor[0, 1, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((imageTensor[0, 2, y, x] / 2 + 0.5), 0, 1) * 255))
                    );
                }
            }

            // Resize image
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(224, 224),
                    Mode = ResizeMode.Crop
                });
            });

            // Preprocess image
            var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };
            for (int y = 0; y < image.Height; y++)
            {
                Span<Rgba32> pixelSpan = image.GetPixelRowSpan(y);

                for (int x = 0; x < image.Width; x++)
                {
                    input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                    input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                    input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                }
            }

            return input;
        }


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        public void Dispose()
        {
            _inferenceSession?.Dispose();
            _sessionOptions?.Dispose();
        }
    }
}

