# Bring the forum up and wait until it's serving. Idempotent.
$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\.."
try {
  docker info *> $null
  if ($LASTEXITCODE -ne 0) { throw "Docker daemon not reachable - is Docker Desktop running (Linux engine)?" }

  if (-not (Test-Path .env)) { Copy-Item .env.example .env }
  docker compose up -d
  if ($LASTEXITCODE -ne 0) { throw "docker compose up failed (exit $LASTEXITCODE)" }
  & "$PSScriptRoot\wait-ready.ps1"

  Write-Host ""
  Write-Host "Reddit is live. From the repo root:"
  Write-Host '  $env:REDDIT = "http://localhost:9999"'
  Write-Host "  python docker/webarena/scripts/capture-auth.py   # REQUIRED: reddit tasks are login-gated"
  Write-Host "  python -m modalitybench.cli run configs/webarena-reddit.yaml"
} finally { Pop-Location }
