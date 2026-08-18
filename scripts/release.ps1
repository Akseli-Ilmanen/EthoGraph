param(
    [string]$Message = "fix: linting"
)

$ErrorActionPreference = "Stop"

pre-commit run --all-files
git add -A
git commit -m $Message
git push origin main

$parts = (git describe --tags --abbrev=0 --match "v*").Split('.')
$parts[-1] = [int]$parts[-1] + 1
$tag = $parts -join '.'
git tag $tag
git push origin $tag

Write-Host "Released $tag"
