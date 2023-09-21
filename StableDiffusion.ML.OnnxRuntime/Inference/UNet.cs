using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using StableDiffusion.ML.OnnxRuntime.Helpers;
using StableDiffusion.ML.OnnxRuntime.Scheduler;

namespace StableDiffusion.ML.OnnxRuntime.Inference
{
    public class UNet : IDisposable
    {
        private readonly SessionOptions _sessionOptions;
        private readonly InferenceSession _inferenceSession;
        private readonly StableDiffusionConfig _configuration;
        private readonly VaeDecoder _vaeDecoder;
        private readonly SafetyChecker _safetyChecker;
        private readonly TextProcessing _textProcessing;
        private readonly PrePackedWeightsContainer _prePackedWeightsContainer;

        /// <summary>
        /// Initializes a new instance of the <see cref="UNet"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public UNet(StableDiffusionConfig configuration)
        {
            _configuration = configuration;
            _sessionOptions = _configuration.GetSessionOptionsForEp();
            _prePackedWeightsContainer = new PrePackedWeightsContainer();
            _inferenceSession = new InferenceSession(_configuration.UnetOnnxPath, _sessionOptions, _prePackedWeightsContainer);
            _vaeDecoder = new VaeDecoder(_configuration, _prePackedWeightsContainer);
            _textProcessing = new TextProcessing(_configuration, _prePackedWeightsContainer);
            if (_configuration.IsSafetyEnabled)
                _safetyChecker = new SafetyChecker(_configuration, _prePackedWeightsContainer);
        }

        public List<NamedOnnxValue> CreateUnetModelInput(Tensor<float> encoderHiddenStates, Tensor<float> sample, long timeStep)
        {

            var input = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor("sample", sample),
                NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<long>(new long[] { timeStep }, new int[] { 1 }))
            };

            return input;

        }


        public Tensor<float> GenerateLatentSample(StableDiffusionConfig config, int seed, float initNoiseSigma)
        {
            return GenerateLatentSample(config.Height, config.Width, seed, initNoiseSigma);
        }


        public Tensor<float> GenerateLatentSample(int height, int width, int seed, float initNoiseSigma)
        {
            var random = new Random(seed);
            var batchSize = 1;
            var channels = 4;
            var latents = new DenseTensor<float>(new[] { batchSize, channels, height / 8, width / 8 });
            var latentsArray = latents.ToArray();

            for (int i = 0; i < latentsArray.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = random.NextDouble(); // Uniform(0,1) random number
                var u2 = random.NextDouble(); // Uniform(0,1) random number
                var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
                var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number

                // add noise to latents with * scheduler.init_noise_sigma
                // generate randoms that are negative and positive
                latentsArray[i] = (float)standardNormalRand * initNoiseSigma;
            }

            latents = TensorHelper.CreateTensor(latentsArray, latents.Dimensions);

            return latents;
        }


        public SixLabors.ImageSharp.Image Inference(string prompt, string negativePrompt = null)
        {
            // Preprocess text
            var textEmbeddings = _textProcessing.PreprocessText(prompt, negativePrompt);

            var scheduler = new LMSDiscreteScheduler();
            //var scheduler = new EulerAncestralDiscreteScheduler();
            var timesteps = scheduler.SetTimesteps(_configuration.NumInferenceSteps);
            //  If you use the same seed, you will get the same image result.
            // var seed = new Random().Next();
            var seed = 329922609;
            Console.WriteLine($"Seed generated: {seed}");

            // create latent tensor
            var latents = GenerateLatentSample(_configuration, seed, scheduler.GetInitNoiseSigma());
            for (int t = 0; t < timesteps.Length; t++)
            {
                // torch.cat([latents] * 2)
                var latentModelInput = TensorHelper.Duplicate(latents, new[] { 2, 4, _configuration.Height / 8, _configuration.Width / 8 });

                // latent_model_input = scheduler.scale_model_input(latent_model_input, timestep = t)
                latentModelInput = scheduler.ScaleInput(latentModelInput, timesteps[t]);

                Console.WriteLine($"scaled model input {latentModelInput[0]} at step {t}. Max {latentModelInput.Max()} Min{latentModelInput.Min()}");
                var input = CreateUnetModelInput(textEmbeddings, latentModelInput, timesteps[t]);

                // Run Inference
                using (var output = _inferenceSession.Run(input))
                {
                    var outputTensor = output.FirstElementAs<DenseTensor<float>>();

                    // Split tensors from 2,4,64,64 to 1,4,64,64
                    var splitTensors = TensorHelper.SplitTensor(outputTensor, new[] { 1, 4, _configuration.Height / 8, _configuration.Width / 8 });
                    var noisePred = splitTensors.Item1;
                    var noisePredText = splitTensors.Item2;

                    // Perform guidance
                    noisePred = PerformGuidance(noisePred, noisePredText, _configuration.GuidanceScale);

                    // LMS Scheduler Step
                    latents = scheduler.Step(noisePred, timesteps[t], latents);
                    Console.WriteLine($"latents result after step {t} min {latents.Min()} max {latents.Max()}");
                }
            }

            // Scale and decode the image latents with vae.
            // latents = 1 / 0.18215 * latents
            latents = TensorHelper.MultipleTensorByFloat(latents, 1.0f / 0.18215f, latents.Dimensions);
            var decoderInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("latent_sample", latents) };

            // Decode image
            var imageResultTensor = _vaeDecoder.Decode(decoderInput);
            if (_configuration.IsSafetyEnabled)
            {
                var isImageSafe = _safetyChecker.IsImageSafe(imageResultTensor);
                if (!isImageSafe)
                    throw new Exception("Resuylting image is NSFW");
            }

            // Convert to image
            return _vaeDecoder.ConvertToImage(imageResultTensor);
        }

        private Tensor<float> PerformGuidance(Tensor<float> noisePred, Tensor<float> noisePredText, double guidanceScale)
        {
            for (int i = 0; i < noisePred.Dimensions[0]; i++)
            {
                for (int j = 0; j < noisePred.Dimensions[1]; j++)
                {
                    for (int k = 0; k < noisePred.Dimensions[2]; k++)
                    {
                        for (int l = 0; l < noisePred.Dimensions[3]; l++)
                        {
                            noisePred[i, j, k, l] = noisePred[i, j, k, l] + (float)guidanceScale * (noisePredText[i, j, k, l] - noisePred[i, j, k, l]);
                        }
                    }
                }
            }
            return noisePred;
        }

        public void Dispose()
        {
            _inferenceSession?.Dispose();
            _sessionOptions?.Dispose();
            _vaeDecoder?.Dispose();
            _safetyChecker?.Dispose();
            _textProcessing?.Dispose();
            _prePackedWeightsContainer?.Dispose();
        }
    }
}
