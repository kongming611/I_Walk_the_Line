$ErrorActionPreference = 'Stop'
# Export-Clixml encrypts SecureStrings with Windows DPAPI for this user/machine.
$values = @{}
$values['DEEPSEEK_API_KEY'] = Read-Host 'DeepSeek API key (hidden)' -AsSecureString
$values['TYPESAFE_API_KEY'] = Read-Host 'Jev / TypeSafe API key (hidden)' -AsSecureString
$values | Export-Clixml -LiteralPath (Join-Path $PSScriptRoot '.credentials.xml')
Write-Host 'Task credentials saved with Windows DPAPI encryption. No API request was sent.'
