#!/usr/bin/env pwsh
# PowerShell script to run multiple SPTAG benchmarks sequentially
# Each benchmark config file will be executed and results saved with timestamp

param(
    [Parameter(Mandatory=$false)]
    [string]$TestExecutable = ".\build\Release\Test.exe",
    
    [Parameter(Mandatory=$false)]
    [string]$ConfigPattern = "benchmark_*.ini",
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = ".\benchmark_results",
    
    [Parameter(Mandatory=$false)]
    [string]$TestCase = "SPFreshTest/BenchmarkFromConfig",

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs = @()
)

# Create output directory if it doesn't exist
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "Created output directory: $OutputDir" -ForegroundColor Green
}

# Find all benchmark config files
$configFiles = Get-ChildItem -Path . -Filter $ConfigPattern | Sort-Object Name

if ($configFiles.Count -eq 0) {
    Write-Host "No benchmark config files found matching pattern: $ConfigPattern" -ForegroundColor Red
    exit 1
}

Write-Host "Found $($configFiles.Count) benchmark config file(s)" -ForegroundColor Cyan
Write-Host "Test Executable: $TestExecutable" -ForegroundColor Cyan
Write-Host "Output Directory: $OutputDir" -ForegroundColor Cyan
Write-Host ""

# Track overall results
$results = @()

foreach ($configFile in $configFiles) {
    # Extract benchmark name (filename without extension)
    $benchmarkName = [System.IO.Path]::GetFileNameWithoutExtension($configFile.Name)
    
    # Generate timestamp (YYYYMMDDHHMM)
    $timestamp = Get-Date -Format "yyyyMMddHHmm"
    
    # Create output filename
    $outputFileName = "${benchmarkName}_${timestamp}.json"
    $outputFilePath = Join-Path $OutputDir $outputFileName
    
    Write-Host "================================================" -ForegroundColor Yellow
    Write-Host "Running benchmark: $benchmarkName" -ForegroundColor Yellow
    Write-Host "Config file: $($configFile.Name)" -ForegroundColor Yellow
    Write-Host "Output file: $outputFileName" -ForegroundColor Yellow
    Write-Host "================================================" -ForegroundColor Yellow
    Write-Host ""
    
    # Set environment variables
    $env:BENCHMARK_CONFIG = $configFile.FullName
    $env:BENCHMARK_OUTPUT = $outputFilePath
    
    # Record start time
    $startTime = Get-Date
    
    # Build argument list (forward extra args so flags like --log_level are passed through)
    $argumentList = @("--run_test=$TestCase")
    if ($ExtraArgs -ne $null -and $ExtraArgs.Count -gt 0) {
        $argumentList += $ExtraArgs
    }

    Write-Host "Running: $TestExecutable $($argumentList -join ' ')" -ForegroundColor Cyan

    # Run the test
    $process = Start-Process -FilePath $TestExecutable `
                             -ArgumentList $argumentList `
                             -NoNewWindow `
                             -Wait `
                             -PassThru
    
    # Record end time
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    # Check exit code
    $exitCode = $process.ExitCode
    $status = if ($exitCode -eq 0) { "SUCCESS" } else { "FAILED" }
    $statusColor = if ($exitCode -eq 0) { "Green" } else { "Red" }
    
    Write-Host ""
    Write-Host "Benchmark '$benchmarkName' completed with status: $status (exit code: $exitCode)" -ForegroundColor $statusColor
    Write-Host "Duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
    Write-Host ""
    
    # Store result
    $results += [PSCustomObject]@{
        BenchmarkName = $benchmarkName
        ConfigFile = $configFile.Name
        OutputFile = $outputFileName
        StartTime = $startTime
        EndTime = $endTime
        Duration = $duration
        ExitCode = $exitCode
        Status = $status
    }
}

# Print summary
Write-Host ""
Write-Host "================================================" -ForegroundColor Magenta
Write-Host "BENCHMARK SUMMARY" -ForegroundColor Magenta
Write-Host "================================================" -ForegroundColor Magenta
Write-Host ""

$results | Format-Table -Property BenchmarkName, Status, Duration, OutputFile -AutoSize

# Count successes and failures
$successCount = ($results | Where-Object { $_.ExitCode -eq 0 }).Count
$failureCount = ($results | Where-Object { $_.ExitCode -ne 0 }).Count

Write-Host ""
Write-Host "Total benchmarks: $($results.Count)" -ForegroundColor Cyan
Write-Host "Successful: $successCount" -ForegroundColor Green
Write-Host "Failed: $failureCount" -ForegroundColor $(if ($failureCount -gt 0) { "Red" } else { "Green" })
Write-Host ""

# Save summary to CSV
$summaryFile = Join-Path $OutputDir "benchmark_summary_$(Get-Date -Format 'yyyyMMddHHmm').csv"
$results | Export-Csv -Path $summaryFile -NoTypeInformation
Write-Host "Summary saved to: $summaryFile" -ForegroundColor Green

# Exit with error if any benchmark failed
if ($failureCount -gt 0) {
    exit 1
}

exit 0
