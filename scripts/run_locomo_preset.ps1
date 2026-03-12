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
            $key = $matches[1].Trim()
            $val = $matches[2].Trim()
            if ($val -ne '') {
                [Environment]::SetEnvironmentVariable($key, $val, 'Process')
            }
        }
    }
}

$maxSamples = 1
$maxQa = 10
$maxQaPerSample = 0
$qaOffset = 0
$sliceFile = ""

switch ($Preset) {
    'quick' {
        $maxSamples = 0
        $maxQa = 0
        $maxQaPerSample = 0
        $sliceFile = 'configs/benchmark/slices/quick.json'
        if (-not $OutRoot) { $OutRoot = 'artifacts/locomo_matrix_preset_quick' }
    }
    'formal' {
        $maxSamples = 0
        $maxQa = 0
        $maxQaPerSample = 0
        $sliceFile = 'configs/benchmark/slices/formal.json'
        if (-not $OutRoot) { $OutRoot = 'artifacts/locomo_matrix_preset_formal' }
    }
    'multisample' {
        $maxSamples = 0
        $maxQa = 0
        $maxQaPerSample = 0
        $sliceFile = 'configs/benchmark/slices/multisample.json'
        if (-not $OutRoot) { $OutRoot = 'artifacts/locomo_matrix_preset_multisample' }
    }
    'advslice' {
        $maxSamples = 0
        $maxQa = 0
        $maxQaPerSample = 0
        $qaOffset = 0
        $sliceFile = 'configs/benchmark/slices/advslice.json'
        if (-not $OutRoot) { $OutRoot = 'artifacts/locomo_matrix_preset_advslice' }
    }
}

if ($sliceFile -and -not (Test-Path $sliceFile)) {
    throw "Missing slice file: $sliceFile. Run scripts/build_locomo_slices.py first."
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
    SliceFile = $sliceFile
    RetrievalTopK = $RetrievalTopK
    OutRoot = $OutRoot
    SkipNli = [bool]$SkipNli
}

Write-Host "[PRESET] $Preset"
Write-Host "[MODEL] provider=$LlmProvider model=$LlmModel"
Write-Host "[RUN] OutRoot=$OutRoot"

& "$PSScriptRoot/run_locomo_matrix.ps1" @matrixArgs
