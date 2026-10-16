$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$appPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $appPython)) {
    Write-Host 'Chua co moi truong .venv. Hay lam theo README_APP.md de cai dat.'
    exit 1
}
& $appPython -m streamlit run app.py --server.address 127.0.0.1
exit $LASTEXITCODE
