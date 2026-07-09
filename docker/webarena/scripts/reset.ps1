# Restore pristine site state between eval runs by recreating the container from the image.
# The Postmill image ships pre-populated, so a fresh container is a fresh, seeded site.
$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\.."
try {
  docker compose down
  if ($LASTEXITCODE -ne 0) { throw "docker compose down failed (exit $LASTEXITCODE)" }
  docker compose up -d --force-recreate
  if ($LASTEXITCODE -ne 0) { throw "docker compose up failed (exit $LASTEXITCODE)" }
  & "$PSScriptRoot\wait-ready.ps1"
  Write-Host "forum reset to pristine state"
} finally { Pop-Location }
