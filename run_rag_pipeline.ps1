param(
    [string]$Question = "How is data stored in milvus?",
    [int]$DockerTimeout = 30,
    [int]$MilvusTimeout = 60,
    [switch]$Stop,
    [switch]$StopDockerDesktop
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Status {
    param([string]$Message)
    Write-Host "$(Get-Date -Format 'HH:mm:ss') | $Message" -ForegroundColor Cyan
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "ERROR: $Message" -ForegroundColor Red
}

function Test-DockerRunning {
    try {
        docker ps 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Start-DockerDesktop {
    Write-Status "Docker Desktop not running. Launching..."

    $dockerPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $dockerPath)) {
        Write-Error-Custom "Docker Desktop not found at $dockerPath"
        exit 1
    }

    & $dockerPath
    Write-Status "Docker Desktop started. Waiting for daemon..."
}

function Wait-DockerDaemon {
    param([int]$TimeoutSeconds = 30)

    $startTime = Get-Date
    $timeout = New-TimeSpan -Seconds $TimeoutSeconds

    while ((Get-Date) - $startTime -lt $timeout) {
        if (Test-DockerRunning) {
            Write-Status "Docker daemon ready"
            return $true
        }
        Start-Sleep -Seconds 2
    }

    Write-Error-Custom "Docker daemon not ready after $TimeoutSeconds seconds"
    return $false
}

function Start-MilvusContainers {
    $composePath = Join-Path $repoRoot "src\milvus-docker\docker-compose.yml"

    if (-not (Test-Path $composePath)) {
        Write-Error-Custom "docker-compose.yml not found at $composePath"
        exit 1
    }

    Write-Status "Starting Milvus containers from $composePath"

    try {
        Push-Location $repoRoot
        & docker compose -f "$composePath" up -d
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "docker compose failed with exit code $LASTEXITCODE"
            exit 1
        }
        Write-Status "Milvus containers started"
    }
    finally {
        Pop-Location
    }
}

function Wait-MilvusHealthy {
    param([int]$TimeoutSeconds = 60)

    $startTime = Get-Date
    $timeout = New-TimeSpan -Seconds $TimeoutSeconds
    $healthUrl = "http://localhost:9091/healthz"

    Write-Status "Polling Milvus health endpoint ($healthUrl)..."

    while ((Get-Date) - $startTime -lt $timeout) {
        try {
            $response = curl -s -w "%{http_code}" -o NUL $healthUrl 2>$null
            if ($response -eq "200") {
                Write-Status "Milvus is healthy"
                return $true
            }
        }
        catch {
            # curl not available, try with PowerShell
            try {
                $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
                if ($response.StatusCode -eq 200) {
                    Write-Status "Milvus is healthy"
                    return $true
                }
            }
            catch {
                # Still waiting
            }
        }
        Start-Sleep -Seconds 3
    }

    Write-Error-Custom "Milvus not healthy after $TimeoutSeconds seconds"
    return $false
}

function Stop-MilvusContainers {
    $composePath = Join-Path $repoRoot "src\milvus-docker\docker-compose.yml"

    if (-not (Test-Path $composePath)) {
        Write-Error-Custom "docker-compose.yml not found at $composePath"
        exit 1
    }

    Write-Status "Stopping Milvus containers from $composePath"

    try {
        Push-Location $repoRoot
        & docker compose -f "$composePath" down
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "docker compose down failed with exit code $LASTEXITCODE"
            exit 1
        }
        Write-Status "Milvus containers stopped and removed"
    }
    finally {
        Pop-Location
    }
}

function Stop-DockerDesktopApp {
    Write-Status "Stopping Docker Desktop..."

    $process = Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue
    if (-not $process) {
        Write-Status "Docker Desktop is not running"
        return
    }

    Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue
    Stop-Process -Name "com.docker.backend" -Force -ErrorAction SilentlyContinue
    Write-Status "Docker Desktop stopped"
}

function Get-MyenvPython {
    $condaBase = (& conda info --base 2>$null).Trim()
    if (-not $condaBase) {
        Write-Error-Custom "Could not resolve conda base path via 'conda info --base'"
        exit 1
    }

    $pythonExe = Join-Path $condaBase "envs\myenv\python.exe"
    if (-not (Test-Path $pythonExe)) {
        Write-Error-Custom "myenv python.exe not found at $pythonExe"
        exit 1
    }

    return $pythonExe
}

function Run-RagPipeline {
    Write-Status "Running RAG pipeline..."

    $pythonExe = Get-MyenvPython

    try {
        Push-Location $repoRoot
        & $pythonExe -m src.main --log-level CRITICAL --question $Question
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Python pipeline failed with exit code $LASTEXITCODE"
            exit 1
        }
        Write-Status "RAG pipeline completed successfully"
    }
    finally {
        Pop-Location
    }
}

# Main execution
try {
    if ($Stop) {
        Write-Host "`nStopping RAG Pipeline`n" -ForegroundColor Green

        Stop-MilvusContainers

        if ($StopDockerDesktop) {
            Stop-DockerDesktopApp
        }

        Write-Host "`nStop completed successfully!`n" -ForegroundColor Green
        exit 0
    }

    Write-Host "`nStarting RAG Pipeline Automation`n" -ForegroundColor Green

    if (-not (Test-DockerRunning)) {
        Start-DockerDesktop
        if (-not (Wait-DockerDaemon -TimeoutSeconds $DockerTimeout)) {
            exit 1
        }
    }
    else {
        Write-Status "Docker daemon already running"
    }

    Start-MilvusContainers

    if (-not (Wait-MilvusHealthy -TimeoutSeconds $MilvusTimeout)) {
        exit 1
    }

    Run-RagPipeline

    Write-Host "`nPipeline completed successfully!`n" -ForegroundColor Green
}
catch {
    Write-Error-Custom $_.Exception.Message
    exit 1
}
