using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using StableDiffusion.ML.OnnxRuntime.Helpers;

namespace StableDiffusion.ML.OnnxRuntime.Inference
{
    public class TextProcessing : IDisposable
    {
        private readonly SessionOptions _sessionOptions;
        private readonly StableDiffusionConfig _configuration;
        private readonly InferenceSession _encoderInferenceSession;
        private readonly InferenceSession _tokenizerInferenceSession;

        /// <summary>
        /// Initializes a new instance of the <see cref="TextProcessing"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public TextProcessing(StableDiffusionConfig configuration, PrePackedWeightsContainer prePackedWeightsContainer)
        {
            _configuration = configuration;
            _sessionOptions = _configuration.GetSessionOptionsForEp();
            _sessionOptions.RegisterOrtExtensions();
            _encoderInferenceSession = new InferenceSession(_configuration.TextEncoderOnnxPath, _sessionOptions, prePackedWeightsContainer);
            _tokenizerInferenceSession = new InferenceSession(_configuration.TokenizerOnnxPath, _sessionOptions, prePackedWeightsContainer);
        }


        /// <summary>
        /// Preprocesses the text.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <returns></returns>
        public DenseTensor<float> PreprocessText(string prompt)
        {
            // Load the tokenizer and text encoder to tokenize and encode the text.
            var textTokenized = TokenizeText(prompt);
            var textPromptEmbeddings = TextEncoder(textTokenized).ToArray();

            // Create uncond_input of blank tokens
            var uncondInputTokens = CreateUncondInput();
            var uncondEmbedding = TextEncoder(uncondInputTokens).ToArray();

            // Concant textEmeddings and uncondEmbedding
            DenseTensor<float> textEmbeddings = new DenseTensor<float>(new[] { 2, 77, 768 });

            for (var i = 0; i < textPromptEmbeddings.Length; i++)
            {
                textEmbeddings[0, i / 768, i % 768] = uncondEmbedding[i];
                textEmbeddings[1, i / 768, i % 768] = textPromptEmbeddings[i];
            }
            return textEmbeddings;
        }


        /// <summary>
        /// Tokenizes the text.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <returns></returns>
        public int[] TokenizeText(string text)
        {
            var inputTensor = new DenseTensor<string>(new string[] { text }, new int[] { 1 });
            var inputString = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("string_input", inputTensor) };

            // Run session and send the input data in to get inference output. 
            using (var tokens = _tokenizerInferenceSession.Run(inputString))
            {
                var inputIds = tokens.FirstElementAs<IEnumerable<long>>();
                Console.WriteLine(string.Join(" ", inputIds));

                // Cast inputIds to Int32
                var InputIdsInt = inputIds.Select(x => (int)x).ToArray();

                var modelMaxLength = 77;
                // Pad array with 49407 until length is modelMaxLength
                if (InputIdsInt.Length < modelMaxLength)
                {
                    var pad = Enumerable.Repeat(49407, 77 - InputIdsInt.Length).ToArray();
                    InputIdsInt = InputIdsInt.Concat(pad).ToArray();
                }

                return InputIdsInt;
            }
        }


        /// <summary>
        ///  Encodes the tokenized input
        /// </summary>
        /// <param name="tokenizedInput">The tokenized input.</param>
        /// <returns></returns>
        public DenseTensor<float> TextEncoder(int[] tokenizedInput)
        {
            // Create input tensor.
            var input_ids = TensorHelper.CreateTensor(tokenizedInput, new[] { 1, tokenizedInput.Count() });
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", input_ids) };

            // Run inference.
            using (var encoded = _encoderInferenceSession.Run(input))
            {
                var lastHiddenState = encoded.FirstElementAs<IEnumerable<float>>();
                var lastHiddenStateTensor = TensorHelper.CreateTensor(lastHiddenState.ToArray(), new[] { 1, 77, 768 });
                return lastHiddenStateTensor;
            }
        }


        /// <summary>
        /// Create an array of empty tokens for the unconditional input.
        /// </summary>
        /// <returns></returns>
        private int[] CreateUncondInput()
        {
            var blankTokenValue = 49407;
            var modelMaxLength = 77;
            var inputIds = new List<int>();
            inputIds.Add(49406);
            var pad = Enumerable.Repeat(blankTokenValue, modelMaxLength - inputIds.Count()).ToArray();
            inputIds.AddRange(pad);

            return inputIds.ToArray();
        }


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        public void Dispose()
        {
            _encoderInferenceSession?.Dispose();
            _tokenizerInferenceSession?.Dispose();
            _sessionOptions?.Dispose();
        }
    }
}