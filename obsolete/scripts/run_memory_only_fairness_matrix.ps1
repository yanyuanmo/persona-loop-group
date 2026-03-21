param(
    [string]$CondaEnv = "persona-loop",
    [string]$Data = "data/locomo10.json",
    [string]$SliceFile = "configs/benchmark/slices/formal.json",
    [int]$MaxQa = 20,
    [string]$LlmProvider = "openai",
    [string]$LlmModel = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    [string]$LlmBaseUrl = "http://127.0.0.1:8080/v1",
    [string]$PersonaMode = "hybrid",
    [string]$PersonaCache = "persona_cache/formal_gpt4o_hybrid.json",
    [string]$OutRoot = "artifacts/memory_only_fairness_formal_q20"
)

$ErrorActionPreference = 'Stop'

$agents = @('persona_loop', 'continuous', 'periodic_remind', 'rag', 'sliding_window', 'ppa')
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($agent in $agents) {
    $outDir = Join-Path $OutRoot $agent
    Write-Host "[RUN] agent=$agent out=$outDir"

    $args = @(
        'run', '--no-capture-output', '-n', $CondaEnv,
        'python', '-u', 'scripts/run_locomo_eval.py',
        '--data', $Data,
        '--agent', $agent,
        '--llm-provider', $LlmProvider,
        '--llm-model', $LlmModel,
        '--llm-base-url', $LlmBaseUrl,
        '--persona-mode', $PersonaMode,
        '--persona-cache', $PersonaCache,
        '--slice-file', $SliceFile,
        '--max-qa', $MaxQa,
        '--eval-mode', 'memory_only',
        '--max-turns', '10',
        '--progress-every', '10',
        '--no-progress-bar',
        '--retrieval-topk', '1',
        '--loop-interval', '7',
        '--loop-retrieval-topk', '1',
        '--loop-recent-turns', '1',
        '--loop-persona-facts', '8',
        '--loop-max-corrections', '2',
        '--persona-risk-filter',
        '--persona-risk-min-llm-confidence', '0.75',
        '--persona-risk-drop-negative',
        '--persona-risk-drop-abstract',
        '--persona-preinject-conflict-gate',
        '--persona-preinject-conflict-scope', 'all',
        '--persona-single-owner-when-ambiguous',
        '--inject-persona-for-all-agents',
        '--output', $outDir
    )

    # LLM endpoint is local OpenAI-compatible; set a dummy key for client validation.
    $env:OPENAI_API_KEY = 'dummy'
    conda @args
}

$rows = @()
foreach ($agent in $agents) {
    $metricPath = Join-Path $OutRoot "$agent/qa_metrics.json"
    if (!(Test-Path $metricPath)) {
        throw "Missing metrics file: $metricPath"
    }
    $m = Get-Content $metricPath -Raw | ConvertFrom-Json
    $rows += [PSCustomObject]@{
        agent = $agent
        f1 = [double]$m.f1
        persona_pcs = [double]$m.persona_pcs
        persona_entailment = [double]$m.persona_entailment
        persona_contradiction = [double]$m.persona_contradiction
        count = [int]$m.count
        inject_persona_for_all_agents = [bool]$m.inject_persona_for_all_agents
    }
}

$rows = $rows | Sort-Object -Property f1 -Descending
$rows | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $OutRoot 'summary.json')
$rows | Export-Csv -Path (Join-Path $OutRoot 'summary.csv') -NoTypeInformation -Encoding UTF8
$rows | Format-Table -AutoSize

Write-Host "[DONE] $OutRoot/summary.json"
