<#
.SYNOPSIS
    Run scripts/bench.py to completion, restarting it if the process dies.

.DESCRIPTION
    bench.py already survives a cell that *raises* — it logs the traceback and
    moves to the next cell. What it cannot survive is the process itself being
    killed (a CUDA OOM the driver turns into a hard abort, the machine running
    out of RAM, a transient that takes the interpreter with it): no Python
    handler runs, so nothing retries. This wrapper is that missing layer.

    Restarting is cheap because bench.py is resumable: a fold with both its
    test evaluation and its prediction set on disk is read back, never
    retrained, so a restart resumes at the fold that died rather than at the
    beginning. Nothing is recomputed.

    Polling ("check every 30 minutes whether it stopped") would do the same job
    worse: it restarts up to 30 minutes late, and it has to guess from the
    outside whether a quiet process is dead or just training. Wrapping the
    process removes the guess — the loop continues the moment bench.py exits.

    Exit codes it reads (bench.py's EXIT_* constants):
      0  every cell finished          -> stop, we are done
      2  nothing finished this run    -> stop, something systemic is wrong
      *  anything else, including a   -> wait, then start it again
         killed process

.PARAMETER MaxRestarts
    Give up after this many starts (default 200). A guard against spinning.

.PARAMETER SleepSeconds
    Wait this long before restarting, so a transient (a file lock, a GPU still
    releasing memory) has time to clear. Default 120.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\bench_watch.ps1
#>
param(
    [int]$MaxRestarts = 200,
    [int]$SleepSeconds = 120
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$bench = Join-Path $repo "scripts\bench.py"

for ($attempt = 1; $attempt -le $MaxRestarts; $attempt++) {
    Write-Host ""
    Write-Host "=== bench attempt $attempt of $MaxRestarts  ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))"
    & python $bench
    $code = $LASTEXITCODE

    if ($code -eq 0) {
        Write-Host "=== every cell finished, after $attempt attempt(s)."
        exit 0
    }
    if ($code -eq 2) {
        Write-Host "=== bench made no progress at all (exit 2) - not restarting."
        Write-Host "=== read the newest log under data\bench_logs\ before trying again."
        exit 2
    }
    Write-Host "=== bench exited $code with cells left; restarting in $SleepSeconds s."
    Start-Sleep -Seconds $SleepSeconds
}

Write-Host "=== gave up after $MaxRestarts attempts; cells remain unfinished."
exit 1
