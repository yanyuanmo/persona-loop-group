param(
    [string]$ModelPath,
    [string]$BinaryPath = "",
    [int]$Port = 8080,
    [int]$ContextSize = 8192,
    [int]$GpuLayers = 99,
    [int]$Threads = 8,
    [string]$HostName = "127.0.0.1"
)

$ErrorActionPreference = 'Stop'

if (-not $ModelPath) {
    throw "ModelPath is required. Example: .\\scripts\\run_local_llamacpp_server.ps1 -ModelPath D:\\models\\qwen2.5-7b-instruct-q4_k_m.gguf"
}

if (-not (Test-Path $ModelPath)) {
    throw "GGUF model file not found: $ModelPath"
}

$candidates = @()
if ($BinaryPath) {
    $candidates += $BinaryPath
}
$candidates += @(
    ".\\llama-server.exe",
    ".\\server.exe",
    ".\\bin\\llama-server.exe",
    ".\\bin\\server.exe"
)

$serverExe = $null
foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path $candidate)) {
        $serverExe = (Resolve-Path $candidate).Path
        break
    }
}

if (-not $serverExe) {
    throw "llama.cpp server executable not found. Provide -BinaryPath or place llama-server.exe under repo/bin/."
}

Write-Host "[llama.cpp] executable: $serverExe"
Write-Host "[llama.cpp] model: $ModelPath"
Write-Host "[llama.cpp] endpoint: http://$HostName`:$Port/v1"
Write-Host "[llama.cpp] Vulkan GPU layers: $GpuLayers"

$argList = @(
    "-m", $ModelPath,
    "--host", $HostName,
    "--port", "$Port",
    "-c", "$ContextSize",
    "-ngl", "$GpuLayers",
    "-t", "$Threads"
)

& $serverExe @argList
