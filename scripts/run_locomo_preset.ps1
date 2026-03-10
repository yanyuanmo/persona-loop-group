param(
    [ValidateSet('quick', 'formal', 'multisample', 'advslice')]
    [string]$Preset = 'quick',
    [string]$CondaEnv = 'persona-loop',
    [string]$Data = 'data/locomo10.json',
    [string]$LlmProvider = 'kimi',
    [string]$LlmModel = 'moonshot-v1-8k',
    [int]$MaxTurns = 50,
    [int]$RetrievalTopK = 3,
    [switch]$SkipNli = $false,
    [string]$OutRoot = ''
)

$ErrorActionPreference = 'Stop'

# Load local env vars (e.g., KIMI_API_KEY) if .env exists.
if (Test-Path '.env') {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
        }
    }
}

$maxSamples = 1
$maxQa = 10
$maxQaPerSample = 0
$qaOffset = 0

switch ($Preset) {
    'quick' {
        $maxSamples = 1
        $maxQa = 10
        $maxQaPerSample = 10
        if (-not $OutRoot) { $OutRoot = 'artifacts/locomo_matrix_preset_quick' }
    }
    'formal' {
        $maxSamples = 1
        $maxQa = 50
        $maxQaPerSample = 50
        if (-not $OutRoot) { $OutRoot = 'artifacts/locomo_matrix_preset_formal' }
    }
    'multisample' {
        $maxSamples = 2
        $maxQa = 40
        $maxQaPerSample = 20
        if (-not $OutRoot) { $OutRoot = 'artifacts/locomo_matrix_preset_multisample' }
    }
    'advslice' {
        $maxSamples = 1
        $maxQa = 20
        $maxQaPerSample = 20
        $qaOffset = 152
        if (-not $OutRoot) { $OutRoot = 'artifacts/locomo_matrix_preset_advslice' }
    }
}

$matrixArgs = @{
    CondaEnv = $CondaEnv
    Data = $Data
    LlmProvider = $LlmProvider
    LlmModel = $LlmModel
    MaxTurns = $MaxTurns
    MaxSamples = $maxSamples
    MaxQa = $maxQa
    MaxQaPerSample = $maxQaPerSample
    QaOffset = $qaOffset
    RetrievalTopK = $RetrievalTopK
    OutRoot = $OutRoot
    SkipNli = [bool]$SkipNli
}

Write-Host "[PRESET] $Preset"
Write-Host "[MODEL] provider=$LlmProvider model=$LlmModel"
Write-Host "[RUN] OutRoot=$OutRoot"

& "$PSScriptRoot/run_locomo_matrix.ps1" @matrixArgs
