# Windows entrypoint: discover the latest nightly, triage, write reports,
# optionally post the markdown to a tracking issue. Mirrors run.sh.
#
# Auth: prefers $env:GITHUB_TOKEN, falls back to `gh auth status`. See
# .env.example for all variables this script reads.
#
# Scheduled Task example (07:15 UTC, after the JAX nightly typically finishes):
#   Action:    powershell.exe
#   Arguments: -NoProfile -ExecutionPolicy Bypass -File C:\path\to\run.ps1
#   Trigger:   Daily at 07:15

#Requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

Set-Location -LiteralPath $PSScriptRoot

# ---- Source a local .env if present (kept out of git) -----------------------
$envFile = Join-Path $PSScriptRoot '.env'
if (Test-Path -LiteralPath $envFile) {
    foreach ($line in Get-Content -LiteralPath $envFile) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) { continue }
        $eq = $trimmed.IndexOf('=')
        if ($eq -lt 1) { continue }
        $name  = $trimmed.Substring(0, $eq).Trim()
        $value = $trimmed.Substring($eq + 1).Trim()
        if ($value.Length -ge 2 -and
            (($value.StartsWith('"') -and $value.EndsWith('"')) -or
             ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        Set-Item -Path "Env:$name" -Value $value
    }
}

# ---- Auth probe -------------------------------------------------------------
# Token wins; otherwise rely on the gh CLI.
$haveToken = (-not [string]::IsNullOrEmpty($env:GITHUB_TOKEN)) -or `
             (-not [string]::IsNullOrEmpty($env:GH_TOKEN))

$haveGhAuth = $false
if (-not $haveToken -and (Get-Command gh -ErrorAction SilentlyContinue)) {
    & gh auth status *> $null
    if ($LASTEXITCODE -eq 0) { $haveGhAuth = $true }
}

if ($haveToken) {
    Write-Host '[run.ps1] using GITHUB_TOKEN env var'
} elseif ($haveGhAuth) {
    Write-Host '[run.ps1] using gh CLI auth'
} else {
    $msg = @'
[run.ps1] ERROR: no GitHub credentials available.

Pick one:
  (1) $env:GITHUB_TOKEN = 'ghp_xxx...'   (https://github.com/settings/tokens)
      For public repos no scope is needed; for private repos use 'repo' or
      a fine-grained token with Actions: read.
  (2) gh auth login                       (https://cli.github.com/)
'@
    [Console]::Error.WriteLine($msg)
    exit 1
}

# ---- Resolve the python interpreter -----------------------------------------
$pythonCmd = Get-Command python  -ErrorAction SilentlyContinue
if (-not $pythonCmd) { $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue }
if (-not $pythonCmd) {
    [Console]::Error.WriteLine('[run.ps1] ERROR: neither python nor python3 found on PATH.')
    exit 1
}
$python = $pythonCmd.Source

# ---- Run the pipeline -------------------------------------------------------
& $python triage.py run @ExtraArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---- Optionally post today's report to a tracking issue ---------------------
$today      = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd')
$reportsDir = if ($env:TRIAGE_REPORTS_DIR) { $env:TRIAGE_REPORTS_DIR } else { 'reports' }
$report     = Join-Path $reportsDir (Join-Path $today 'report.md')

if ((Test-Path -LiteralPath $report) -and -not [string]::IsNullOrEmpty($env:TRACKING_ISSUE)) {
    Write-Host "[run.ps1] posting report to $($env:TRACKING_ISSUE)..."
    $hashIdx = $env:TRACKING_ISSUE.LastIndexOf('#')
    if ($hashIdx -lt 1 -or $hashIdx -ge ($env:TRACKING_ISSUE.Length - 1)) {
        Write-Warning "[run.ps1] TRACKING_ISSUE must be of the form 'owner/repo#NUM'"
    } else {
        $repoPart = $env:TRACKING_ISSUE.Substring(0, $hashIdx)
        $issueNum = $env:TRACKING_ISSUE.Substring($hashIdx + 1)

        if (Get-Command gh -ErrorAction SilentlyContinue) {
            & gh issue comment $issueNum --repo $repoPart --body-file $report
        } else {
            $token = if ($env:GITHUB_TOKEN) { $env:GITHUB_TOKEN } else { $env:GH_TOKEN }
            if ([string]::IsNullOrEmpty($token)) {
                Write-Warning '[run.ps1] cannot post comment: no token and no gh CLI'
            } else {
                $body    = Get-Content -LiteralPath $report -Raw
                $payload = @{ body = $body } | ConvertTo-Json -Compress
                $headers = @{
                    'Authorization'        = "Bearer $token"
                    'Accept'               = 'application/vnd.github+json'
                    'X-GitHub-Api-Version' = '2022-11-28'
                    'User-Agent'           = 'jax-nightly-triage/1.0'
                }
                $resp = Invoke-RestMethod `
                    -Uri "https://api.github.com/repos/$repoPart/issues/$issueNum/comments" `
                    -Method Post -Headers $headers -Body $payload `
                    -ContentType 'application/json'
                Write-Host $resp.html_url
            }
        }
    }
}

# ---- Hygiene: keep only the last 60 days of report dirs ---------------------
$cutoff = (Get-Date).AddDays(-60)
if (Test-Path -LiteralPath $reportsDir) {
    Get-ChildItem -LiteralPath $reportsDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        ForEach-Object {
            try {
                Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop
            } catch {
                Write-Warning "[run.ps1] could not remove $($_.FullName): $($_.Exception.Message)"
            }
        }
}
