param(
    [string]$BaseUrl = "http://127.0.0.1:8080/v1",
    [string]$ApiKey = "dummy",
    [string]$Model = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    [string]$Sizes = "2000,8000,16000,32000,64000",
    [int]$MaxTokens = 16,
    [string]$Output = "artifacts/local_profile"
)

$ErrorActionPreference = 'Stop'

$env:OPENAI_BASE_URL = $BaseUrl
$env:OPENAI_API_KEY = $ApiKey
$env:LOCAL_MODEL_NAME = $Model

$args = @(
    'run', '-n', 'persona-loop',
    'python', 'scripts/profile_local_llm.py',
    '--base-url', $BaseUrl,
    '--api-key', $ApiKey,
    '--model', $Model,
    '--sizes', $Sizes,
    '--max-tokens', "$MaxTokens",
    '--output', $Output
)

conda @args
