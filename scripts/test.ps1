# RetrievalHub 测试与运行脚本 (PowerShell)
# 用法: .\scripts\test.ps1 <command>

param(
    [Parameter(Position=0)]
    [string]$Command = "all"
)

$ErrorActionPreference = "Stop"

function Invoke-UnitTests {
    Write-Host "`n[1/4] 单元测试..." -ForegroundColor Cyan
    pytest tests/test_phase0.py -v --tb=short
}

function Invoke-Coverage {
    Write-Host "`n[2/4] 覆盖率报告..." -ForegroundColor Cyan
    pytest tests/ --cov=retrievalhub --cov-report=term-missing --cov-report=html
}

function Invoke-Integration {
    Write-Host "`n[3/4] 集成测试..." -ForegroundColor Cyan
    pytest tests/ -v -k "integration or e2e" --tb=short
}

function Invoke-All {
    Write-Host "`n========== RetrievalHub 全量测试 ==========" -ForegroundColor Yellow
    Invoke-UnitTests
    Invoke-Coverage
    Write-Host "`n[3/4] 集成测试 (暂无，跳过)..." -ForegroundColor Cyan
    Write-Host "`n[4/4] 端到端测试 (暂无，跳过)..." -ForegroundColor Cyan
    Write-Host "`n========== 全部测试通过 ==========" -ForegroundColor Green
}

switch ($Command) {
    "unit" { Invoke-UnitTests }
    "cov" { Invoke-Coverage }
    "integration" { Invoke-Integration }
    "all" { Invoke-All }
    default {
        Write-Host "用法: .\scripts\test.ps1 [unit|cov|integration|all]"
        Write-Host "  unit       - 仅运行单元测试"
        Write-Host "  cov        - 运行测试并生成覆盖率报告"
        Write-Host "  integration- 运行集成测试"
        Write-Host "  all        - 运行全部测试（默认）"
    }
}
