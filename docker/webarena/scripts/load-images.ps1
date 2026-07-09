# Load the downloaded forum tar into Docker as the image the compose file references.
$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\.."
try {
  $tar = "downloads\postmill-populated-exposed-withimg.tar"
  if (-not (Test-Path $tar)) { throw "Missing $tar - run download-images.ps1 first" }

  # Preflight: native exes don't trip $ErrorActionPreference, so check the daemon explicitly.
  docker info *> $null
  if ($LASTEXITCODE -ne 0) { throw "Docker daemon not reachable - is Docker Desktop running (Linux engine)?" }

  Write-Host "docker load < $tar (this unpacks several GB) ..."
  docker load --input $tar
  if ($LASTEXITCODE -ne 0) { throw "docker load failed (exit $LASTEXITCODE)" }
  Write-Host "Loaded. Image: postmill-populated-exposed-withimg"
} finally { Pop-Location }
