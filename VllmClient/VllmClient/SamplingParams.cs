namespace VllmClient;

public enum EarlyStopping
{
    Heuristic, // False
    BestOf, // True
    Never // "never"
}

public class SamplingParams
{
    private const float EPS = 1e-5f;

    public int N { get; set; } = 1;
    public int? BestOf { get; set; }
    public float PresencePenalty { get; set; } = 0f;
    public float FrequencyPenalty { get; set; } = 0f;
    public float Temperature { get; set; } = 1f;
    public float TopP { get; set; } = 1f;
    public int TopK { get; set; } = -1;
    public bool UseBeamSearch { get; set; }
    public float LengthPenalty { get; set; } = 1f;
    public EarlyStopping EarlyStopping { get; set; } = EarlyStopping.Heuristic;
    public IList<string>? Stop { get; set; }
    public IList<int>? StopTokenIds { get; set; }
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