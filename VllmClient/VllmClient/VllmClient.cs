using System.Text.Json;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Unicode;

namespace VllmClient;

class VllmClientException : Exception
{
    public VllmClientException(string message) : base(message)
    {
    }
}

public class AsyncVllmClient : IDisposable
{
    private readonly HttpClient client;

    public AsyncVllmClient(string apiUrl)
    {
        client = new HttpClient { BaseAddress = new Uri(apiUrl) };
    }

    public void Dispose()
    {
        client.Dispose();
    }

    public async Task<IList<string>> Generate(string prompt, SamplingParams @params, CancellationToken cancellationToken = default)
    {
        var payload = FormatRequestData(prompt, false, @params);

        var response = await client.PostAsJsonAsync("", payload, cancellationToken: cancellationToken);
        response.EnsureSuccessStatusCode();

        var data = await response.Content.ReadFromJsonAsync<Dictionary<string, JsonElement>?>(cancellationToken: cancellationToken);
        if (data == null)
        {
            throw new VllmClientException("Invalid server response (not JSON)");
        }

        if (!data.TryGetValue("text", out var texts))
        {
            throw new VllmClientException("Invalid server response (does not have a \"text\" item)");
        }

        if (texts.ValueKind != JsonValueKind.Array)
        {
            throw new VllmClientException("Invalid server response (\"text\" item is not an array)");
        }

        return texts.EnumerateArray().Select(v => v.GetString() ?? "N/A").ToList();
    }

    public async IAsyncEnumerable<IList<string>> Stream(string prompt, SamplingParams @params, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var payload = FormatRequestData(prompt, true, @params);

        var response = await client.PostAsJsonAsync("", payload, cancellationToken: cancellationToken);
        var content = await response.Content.ReadAsStreamAsync(cancellationToken);

        var buffer = new byte[32768];
        var filled = 0;

        for (;;)
        {
            var bytesRead = await content.ReadAsync(buffer.AsMemory(filled), cancellationToken);
            if (bytesRead == 0)
            {
                if (filled > 0)
                {
                    throw new VllmClientException("Unexpected end of stream");
                }

                break;
            }

            filled += bytesRead;

            for (;;)
            {
                var zero = Array.FindIndex(buffer, 0, filled, b => b == 0);
                if (zero < 0)
                {
                    if (filled == buffer.Length)
                    {
                        Array.Resize(ref buffer, buffer.Length * 2);
                    }

                    break;
                }

                var jsonDoc = JsonDocument.Parse(buffer.AsMemory(0, zero));
                var textItem = jsonDoc.RootElement.GetProperty("text");
                if (textItem.ValueKind != JsonValueKind.Array)
                {
                    throw new VllmClientException("Invalid server response");
                }

                var texts = textItem.EnumerateArray().Select(v => v.GetString() ?? "N/A").ToList();
                yield return texts;

                var consumed = zero + 1;
                if (filled > consumed)
                {
                    buffer.AsSpan(consumed).CopyTo(buffer);
                }

                filled -= consumed;
            }
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
            { "stop", @params.Stop == null ? null : @params.Stop.Count == 1 ? @params.Stop[0] : @params.Stop },
            { "stop_token_ids", @params.StopTokenIds },
            { "ignore_eos", @params.IgnoreEos },
            { "max_tokens", @params.MaxTokens },
            { "logprobs", @params.Logprobs },
            { "skip_special_tokens", @params.SkipSpecialTokens },
        };


        payload["early_stopping"] = @params.EarlyStopping switch
        {
            EarlyStopping.Heuristic => false,
            EarlyStopping.BestOf => true,
            EarlyStopping.Never => "never",
            _ => null
        };

        return payload;
    }
}