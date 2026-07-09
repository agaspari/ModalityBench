# Poll the forum until it answers, so callers don't race the container's startup.
param([string]$Url)
if (-not $Url) {
  if ($env:REDDIT_PORT) { $port = $env:REDDIT_PORT } else { $port = "9999" }
  $Url = "http://localhost:$port"
}

Write-Host "Waiting for forum at $Url ..."
for ($i = 0; $i -lt 60; $i++) {
  $code = & curl.exe -s -o NUL -w "%{http_code}" $Url
  if ($code -eq "200" -or $code -eq "301" -or $code -eq "302") {
    Write-Host "forum is up (HTTP $code)"; return
  }
  Start-Sleep -Seconds 2
}
throw "forum did not become ready within ~2 min"
