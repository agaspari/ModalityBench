# Download WebArena's raw task config (all 812 tasks; WebArenaSource filters to sites: [reddit]
# and substitutes __REDDIT__ from env, so no generate_test_data.py needed).
$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\.."
try {
  New-Item -ItemType Directory -Force config_files | Out-Null
  $url = "https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files/test.raw.json"
  curl.exe -L -o "config_files\test.raw.json" $url
  Write-Host "saved -> config_files\test.raw.json"
} finally { Pop-Location }
