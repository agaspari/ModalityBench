# Download the Reddit (Postmill) image tar from WebArena's CMU mirror (no Google Drive gate).
# Resumable (curl -C -): re-run if interrupted. A few GB. Run from anywhere.
$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\.."
try {
  $tar = "postmill-populated-exposed-withimg.tar"
  if ($env:FORUM_URL) { $url = $env:FORUM_URL } else { $url = "http://metis.lti.cs.cmu.edu/webarena-images/$tar" }

  New-Item -ItemType Directory -Force downloads | Out-Null
  Write-Host "Downloading $tar from $url"
  curl.exe -L -C - -o "downloads\$tar" $url
  if ($LASTEXITCODE -ne 0) { throw "download failed (curl exit $LASTEXITCODE) - re-run to resume" }
  Write-Host "Done -> downloads\$tar"
} finally { Pop-Location }
