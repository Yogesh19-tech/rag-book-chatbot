# =====================================================
# install.ps1
# Run this script to install all project dependencies
# into the virtual environment.
#
# HOW TO RUN:
#   1. Open PowerShell
#   2. Navigate to project folder:
#      cd "C:\Users\SRIYOGI KALAI\OneDrive\Desktop\book_rag_chatbot"
#   3. Run:
#      .\install.ps1
# =====================================================

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  Book RAG Chatbot - Package Installer" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

$pip = ".\venv\Scripts\pip.exe"

# Check venv exists
if (-Not (Test-Path $pip)) {
    Write-Host "[STEP 1] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "[OK] Virtual environment already exists." -ForegroundColor Green
}

Write-Host ""
Write-Host "[STEP 2] Upgrading pip..." -ForegroundColor Yellow
& $pip install --upgrade pip
Write-Host ""

# Install each library one by one with clear output
$packages = @(
    "Flask==3.0.3",
    "werkzeug==3.0.3",
    "python-dotenv==1.0.1",
    "mysql-connector-python==8.4.0",
    "Pillow==10.4.0",
    "chromadb==0.5.3",
    "sentence-transformers==3.0.1",
    "transformers==4.43.3",
    "accelerate==0.33.0",
    "torch==2.3.1"
)

$total = $packages.Count
$i = 1

foreach ($pkg in $packages) {
    Write-Host "[$i/$total] Installing $pkg ..." -ForegroundColor Yellow
    & $pip install $pkg
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] $pkg installed." -ForegroundColor Green
    } else {
        Write-Host "[WARN] $pkg may have had an issue. Check output above." -ForegroundColor Red
    }
    Write-Host ""
    $i++
}

Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  All packages installation complete!" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the virtual environment, run:" -ForegroundColor White
Write-Host "  .\venv\Scripts\activate" -ForegroundColor Yellow
Write-Host ""
Write-Host "To start the app, run:" -ForegroundColor White
Write-Host "  python app.py" -ForegroundColor Yellow
Write-Host ""
