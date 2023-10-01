using System.Text.Json;
using System.Net.Http.Json;
using System.Text;

namespace VllmClient;

public static class Version
{
    public const string Current = "0.2.0";
}

public enum EarlyStopping
{
    Heuristic, // False
    BestOf, // True
    Never // "never"
}

public class SamplingParams
{
    public const float EPS = 1e-5f;

    public int N { get; set; } = 1;
    public int? BestOf { get; set; }
    public float PresencePenalty { get; set; } = 0;
    public float FrequencyPenalty { get; set; } = 0;
    public float Temperature { get; set; } = 1;
    public float TopP { get; set; } = 1;
    public int TopK { get; set; } = -1;
    public bool UseBeamSearch { get; set; }
    public float LengthPenalty { get; set; } = 1;
    public EarlyStopping EarlyStopping { get; set; } = EarlyStopping.BestOf;
    public IList<string> Stop { get; set; }
    public IList<int> StopTokenIds { get; set; }
    public bool IgnoreEos { get; set; }
    public int MaxTokens { get; set; } = 16;
    public int? Logprobs { get; set; }
    public bool SkipSpecialTokens { get; set; } = true;

    public SamplingParams()
    {
        Validate();
    }

    private void Validate()
    {
        if (N < 1)
        {
            throw new ArgumentException("N must be at least 1");
        }

        if (BestOf < N)
        {
            throw new ArgumentException("BestOf must be >= N");
        }

        if (PresencePenalty < -2 || PresencePenalty > 2)
        {
            throw new ArgumentException("PresencePenalty must be in [-2, 2]");
        }

        // Remaining validation checks
    }

    private void ValidateBeamSearch()
    {
        if (BestOf <= 1)
        {
            throw new InvalidOperationException("BestOf must be > 1 for beam search");
        }

        if (Temperature > EPS)
        {
            throw new InvalidOperationException("Temp must be 0 for beam search");
        }

        // Remaining beam search checks
    }

    private void ValidateNonBeamSearch()
    {
        if (EarlyStopping != EarlyStopping.Heuristic)
        {
            throw new InvalidOperationException("Early stopping must be Heuristic without beam search");
        }

        if (LengthPenalty < 1 - EPS || LengthPenalty > 1 + EPS)
        {
            throw new InvalidOperationException("Length penalty must be 1 without beam search");
        }
    }

    private void ValidateGreedySampling()
    {
        if (BestOf > 1)
        {
            throw new InvalidOperationException("BestOf must be 1 for greedy sampling");
        }

        if (TopP < 1 - EPS)
        {
            throw new InvalidOperationException("TopP must be 1 for greedy sampling");
        }

        if (TopK != -1)
        {
            throw new InvalidOperationException("TopK must be -1 for greedy sampling");
        }
    }
}

public class AsyncVllmClient
{
    private readonly HttpClient client;

    public AsyncVllmClient(string url)
    {
        client = new HttpClient { BaseAddress = new Uri(url) };
    }

    public async Task<List<string>> Generate(string prompt, SamplingParams @params)
    {
        var payload = FormatRequestData(prompt, false, @params);

        var response = await client.PostAsJsonAsync("", payload);
        response.EnsureSuccessStatusCode();

        return await response.Content.ReadFromJsonAsync<List<string>>() ?? new List<string>();
    }

    public async IAsyncEnumerable<List<string>> Stream(string prompt, SamplingParams @params)
    {
        var payload = FormatRequestData(prompt, true, @params);

        var response = await client.PostAsJsonAsync("", payload);
        using var request = new HttpRequestMessage(HttpMethod.Post, "");
        request.Content = JsonContent.Create(payload);

        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync();

        while (true)
        {
            var delimitedJson = await ReadDelimitedJsonAsync(stream);
            if (delimitedJson == null)
                break;

            yield return JsonSerializer.Deserialize<List<string>>(delimitedJson);
        }
    }

    private static Dictionary<string, object?> FormatRequestData(string prompt, bool stream, SamplingParams @params)
    {
        var payload = new Dictionary<string, object?>()
        {
            { "prompt", prompt },
            { "stream", stream },
            { "n", @params.N },
            { "best_of", @params.BestOf },
            { "presence_penalty", @params.PresencePenalty },
            { "frequency_penalty", @params.FrequencyPenalty },
            { "temperature", @params.Temperature },
            { "top_p", @params.TopP },
            { "top_k", @params.TopK },
            { "use_beam_search", @params.UseBeamSearch },
            { "length_penalty", @params.LengthPenalty },
            { "early_stopping", @params.EarlyStopping },
            { "stop", @params.Stop },
            { "stop_token_ids", @params.StopTokenIds },
            { "ignore_eos", @params.IgnoreEos },
            { "max_tokens", @params.MaxTokens },
            { "logprobs", @params.Logprobs },
            { "skip_special_tokens", @params.SkipSpecialTokens },
        };
        return payload;
    }

    private async Task<string> ReadDelimitedJsonAsync(Stream stream)
    {
        var buffer = new byte[1024];
        var totalLength = 0;
        var delimitersFound = 0;

        while (true)
        {
            var bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length);
            if (bytesRead == 0) return null;

            for (var i = 0; i < bytesRead; i++)
            {
                if (buffer[i] == 0) delimitersFound++;
                else delimitersFound = 0;

                if (delimitersFound == 2)
                {
                    var partialJson = Encoding.UTF8.GetString(buffer, 0, i + 1);
                    return partialJson;
                }
            }

            totalLength += bytesRead;
            if (totalLength > 1024 * 10)
            {
                // 10 KB
                throw new Exception("Exceeded maximum size per item");
            }
        }
    }
}