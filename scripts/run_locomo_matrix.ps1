param(
    [string]$CondaEnv = "persona-loop",
    [string]$Data = "data/locomo10.json",
    [string]$LlmProvider = "hf",
    [string]$LlmModel = "Qwen/Qwen2.5-0.5B-Instruct",
    [int]$MaxTurns = 50,
    [int]$MaxSamples = 1,
    [int]$MaxQa = 20,
    [int]$MaxQaPerSample = 0,
    [int]$QaOffset = 0,
    [string]$SliceFile = "",
    [int]$RetrievalTopK = 3,
    [switch]$SkipNli = $true,
    [string]$OutRoot = "artifacts/locomo_matrix_quick"
)

$ErrorActionPreference = 'Stop'

$agents = @('continuous', 'rag', 'persona_loop')
$modes = @('open_book', 'hide_evidence', 'memory_only')

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$rows = @()
$runStartedAt = (Get-Date).ToUniversalTime().ToString("o")
$gitCommit = "unknown"
try {
    $gitCommit = (git rev-parse HEAD).Trim()
} catch {
    $gitCommit = "unknown"
}

foreach ($agent in $agents) {
    foreach ($mode in $modes) {
        $runName = "{0}_{1}" -f $agent, $mode
        $outDir = Join-Path $OutRoot $runName

        $args = @(
            'run', '-n', $CondaEnv,
            'python', 'scripts/run_locomo_eval.py',
            '--data', $Data,
            '--agent', $agent,
            '--llm-provider', $LlmProvider,
            '--llm-model', $LlmModel,
            '--max-turns', $MaxTurns,
            '--max-samples', $MaxSamples,
            '--max-qa', $MaxQa,
            '--max-qa-per-sample', $MaxQaPerSample,
            '--qa-offset', $QaOffset,
            '--eval-mode', $mode,
            '--retrieval-topk', $RetrievalTopK,
            '--output', $outDir
        )

        if ($SliceFile) {
            $args += @('--slice-file', $SliceFile)
        }

        if ($SkipNli) {
            $args += '--skip-nli'
        }

        Write-Host "[RUN] agent=$agent mode=$mode out=$outDir"
        conda @args

        $metricPath = Join-Path $outDir 'qa_metrics.json'
        if (!(Test-Path $metricPath)) {
            throw "Missing metrics file: $metricPath"
        }

        $m = Get-Content $metricPath -Raw | ConvertFrom-Json
        $rows += [PSCustomObject]@{
            agent = $agent
            eval_mode = $mode
            count = $m.count
            em = $m.em
            f1 = $m.f1
            nli_entailment_gold = $m.nli_entailment_gold
            nli_contradiction_gold = $m.nli_contradiction_gold
            nli_entailment_adv = $m.nli_entailment_adv
            nli_contradiction_adv = $m.nli_contradiction_adv
            persona_pcs = $m.persona_pcs
            evidence_visible_ratio = $m.evidence_visible_ratio
            history_items_avg = $m.history_items_avg
            adversarial_count = $m.adversarial_count
        }
    }
}

$csvPath = Join-Path $OutRoot 'summary.csv'
$jsonPath = Join-Path $OutRoot 'summary.json'

$rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
$rows | ConvertTo-Json -Depth 5 | Set-Content -Path $jsonPath -Encoding UTF8

$manifest = [PSCustomObject]@{
    created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    started_at_utc = $runStartedAt
    git_commit = $gitCommit
    script = "scripts/run_locomo_matrix.ps1"
    args = [PSCustomObject]@{
        CondaEnv = $CondaEnv
        Data = $Data
        LlmProvider = $LlmProvider
        LlmModel = $LlmModel
        MaxTurns = $MaxTurns
        MaxSamples = $MaxSamples
        MaxQa = $MaxQa
        MaxQaPerSample = $MaxQaPerSample
        QaOffset = $QaOffset
        SliceFile = $SliceFile
        RetrievalTopK = $RetrievalTopK
        SkipNli = [bool]$SkipNli
        OutRoot = $OutRoot
    }
    summary_csv = $csvPath
    summary_json = $jsonPath
    runs = $rows
}

$manifestPath = Join-Path $OutRoot 'run_manifest.json'
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host "[DONE] Summary CSV: $csvPath"
Write-Host "[DONE] Summary JSON: $jsonPath"
Write-Host "[DONE] Manifest: $manifestPath"
