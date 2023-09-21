using StableDiffusion.ML.OnnxRuntime.Inference;
using System.Diagnostics;

namespace StableDiffusion
{
    public class Program
    {
        static void Main(string[] _)
        {
            //test how long this takes to execute
            var timestamp = Stopwatch.GetTimestamp();

            //Default args
            var prompt = "a fireplace in an old cabin in the woods";
            var negativePrompt = "stone, chair";

            Console.WriteLine($"Prompt: {prompt}");
            Console.WriteLine($"NegativePrompt: {negativePrompt}");

            var config = new StableDiffusionConfig
            {
                // The number of steps to run inference for.
                // The more steps the longer it will take to run the inference loop but the image quality should improve.
                NumInferenceSteps = 15,

                // The scale for the classifier-free guidance.
                // The higher the number the more it will try to look like the prompt but the image quality may suffer.
                GuidanceScale = 7.5,

                // Set your preferred Execution Provider. Currently (GPU, DirectML, CPU) are supported in this project.
                // ONNX Runtime supports many more than this. Learn more here: https://onnxruntime.ai/docs/execution-providers/
                // The config is defaulted to CUDA. You can override it here if needed.
                // To use DirectML EP intall the Microsoft.ML.OnnxRuntime.DirectML and uninstall Microsoft.ML.OnnxRuntime.GPU
                ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.DirectML,

                // Set GPU Device ID.
                DeviceId = 0,

                // Update paths to your models
                TextEncoderOnnxPath = @"D:\Repositories\stable-diffusion-v1-5\text_encoder\model.onnx",
                UnetOnnxPath = @"D:\Repositories\stable-diffusion-v1-5\unet\model.onnx",
                VaeDecoderOnnxPath = @"D:\Repositories\stable-diffusion-v1-5\vae_decoder\model.onnx",
                SafetyModelPath = @"D:\Repositories\stable-diffusion-v1-5\safety_checker\model.onnx",

                // Is Anti-NSFW Enabled
                IsSafetyEnabled = false
            };

            // Inference Stable Diff
            using (var uNet = new UNet(config))
            {
                var image = uNet.Inference(prompt, negativePrompt);

                // If image failed or was unsafe it will return null.
                if (image == null)
                {
                    Console.WriteLine("Unable to create image, please try again.");
                }
            }

            // Stop the timer
            Console.WriteLine($"Time taken: {Stopwatch.GetElapsedTime(timestamp).TotalMilliseconds}ms");
        }
    }
}