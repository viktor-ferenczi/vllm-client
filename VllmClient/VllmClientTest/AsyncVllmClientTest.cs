namespace VllmClientTest;

public class AsyncVllmClientTest
{
    private string ApiUrl => Environment.GetEnvironmentVariable("VLLM_BASE_URL") ?? "http://localhost:8000";
    private const string Prompt = "This is an inteview with a biologist. Reporter: How many fingers humans have on one hand? Biologist: ";

    private AsyncVllmClient? client;

    [SetUp]
    public void Setup()
    {
        client = new AsyncVllmClient(ApiUrl);
    }

    [TearDown]
    public void Teardown()
    {
        client?.Dispose();
        client = null;
    }

    [Test]
    public async Task TestSimpleGeneration()
    {
        if (client == null)
        {
            Assert.Fail();
            return;
        }

        var @params = new SamplingParams { Temperature = 0.01f, MaxTokens = 2 };

        var response = await client.Generate(Prompt, @params);
        Assert.That(response, Has.Count.EqualTo(1));

        var output = response[0];
        Assert.That(output.Contains('5') || output.ToLower().Contains("five"), $"Wrong output: {output}");
    }

    [Test]
    public async Task TestStreamingGeneration()
    {
        if (client == null)
        {
            Assert.Fail();
            return;
        }

        var parts = new List<string>();
        var received = Prompt.Length;

        var @params = new SamplingParams { Temperature = 0.01f, MaxTokens = 2 };
        await foreach (var response in client.Stream(Prompt, @params))
        {
            Assert.That(response, Has.Count.EqualTo(1));
            var part = response[0][received..];

            parts.Add(part);
            received += part.Length;
        }

        var output = string.Join("", parts);
        Assert.That(output.Contains('5') || output.ToLower().Contains("five"), $"Wrong output: {output}");
    }
}
