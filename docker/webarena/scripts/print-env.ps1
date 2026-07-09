# Dot-source to set REDDIT in your current shell:
#   . docker/webarena/scripts/print-env.ps1
if (-not $env:REDDIT) {
  if ($env:REDDIT_PORT) { $port = $env:REDDIT_PORT } else { $port = "9999" }
  $env:REDDIT = "http://localhost:$port"
}
Write-Host "REDDIT=$($env:REDDIT)"
