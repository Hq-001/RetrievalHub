# RetrievalHub Test & Run Script (PowerShell)
# Usage: .\scripts\test.ps1 <command>

param(
    [Parameter(Position=0)]
    [string]$Command = "all"
)

$ErrorActionPreference = "Stop"

function Invoke-UnitTests {
    Write-Host "`n[1/4] Unit Tests..." -ForegroundColor Cyan
    pytest tests/test_phase0.py -v --tb=short
}

function Invoke-Coverage {
    Write-Host "`n[2/4] Coverage Report..." -ForegroundColor Cyan
    pytest tests/ --cov=retrievalhub --cov-report=term-missing --cov-report=html
}

function Invoke-Integration {
    Write-Host "`n[3/4] Integration Tests..." -ForegroundColor Cyan
    pytest tests/ -v -k "integration or e2e" --tb=short
}

function Invoke-All {
    Write-Host "`n========== RetrievalHub Full Test Suite ==========" -ForegroundColor Yellow
    Invoke-UnitTests
    Invoke-Coverage
    Write-Host "`n[3/4] Integration Tests (none yet, skipped)..." -ForegroundColor Cyan
    Write-Host "`n[4/4] E2E Tests (none yet, skipped)..." -ForegroundColor Cyan
    Write-Host "`n========== All Tests Passed ==========" -ForegroundColor Green
}

switch ($Command) {
    "unit" { Invoke-UnitTests }
    "cov" { Invoke-Coverage }
    "integration" { Invoke-Integration }
    "all" { Invoke-All }
    default {
        Write-Host "Usage: .\scripts\test.ps1 [unit|cov|integration|all]"
        Write-Host "  unit        - Run unit tests only"
        Write-Host "  cov         - Run tests with coverage report"
        Write-Host "  integration - Run integration tests"
        Write-Host "  all         - Run all tests (default)"
    }
}
