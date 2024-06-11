using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Text;
using System.Numerics.Tensors;
using System.Text;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddKernel().AddOpenAIChatCompletion("gpt-4", builder.Configuration["AI:OpenAI:ApiKey"])
    .AddOpenAITextEmbeddingGeneration("text-embedding-3-small", builder.Configuration["AI:OpenAI:ApiKey"]);


var app = builder.Build();

var code = File.ReadAllLines(@"TensorPrimitives.netcore.cs");

var tokenizer = await Tiktoken.CreateByModelNameAsync("gpt-4");

var chunks = TextChunker.SplitPlainTextParagraphs([.. code], 500, 100, null, text => tokenizer.CountTokens(text));

List<(string Content, ReadOnlyMemory<float> Vector)> db =
    chunks.Zip(await app.Services.GetRequiredService<ITextEmbeddingGenerationService>().GenerateEmbeddingsAsync(chunks)).ToList();


app.MapGet("/copilot", async (string question, Kernel kernel, ITextEmbeddingGenerationService embeddingService) =>
{
    var qe = await embeddingService.GenerateEmbeddingAsync(question);

    var prompt = new StringBuilder("Please answer this question: ")
    .AppendLine(question)
    .AppendLine("*** Code: ");

    int tokensRemaining = 2000;

    foreach(var c in db.OrderByDescending(c => TensorPrimitives.CosineSimilarity<float>(qe.Span, c.Vector.Span)))
    {

        if ((tokensRemaining -= tokenizer.CountTokens(c.Content)) < 0)
            break;

        prompt.AppendLine(c.Content);
    }

    return kernel.InvokePromptStreamingAsync<string>(prompt.ToString());
});

app.Run();
