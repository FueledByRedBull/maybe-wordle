#requires -Version 7.0

[CmdletBinding()]
param(
    [switch] $Check
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$redactedWord = '[redacted]'
$markerName = 'public_redaction'
$markerKind = 'maybe-wordle-public-evidence-redaction-v1'
$markerVersion = 1
$markerFields = @('game.target', 'game.path[]')
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

$specs = @(
    [pscustomobject]@{
        Name = 'rolling'
        Schema = 4
        Input = Join-Path $repoRoot 'benchmarks/predictive/september-finite-preordered-v1.json'
        Output = Join-Path $repoRoot 'docs/evidence/september-finite-preordered-public-v1.json'
    },
    [pscustomobject]@{
        Name = 'benchmark'
        Schema = 7
        Input = Join-Path $repoRoot 'benchmarks/predictive/september-post-layout-tests-rolling-v1.json'
        Output = Join-Path $repoRoot 'docs/evidence/september-post-layout-tests-rolling-public-v1.json'
    }
)

function Stop-Redaction([string] $Message) {
    throw "public evidence redaction: $Message"
}

function Get-PropertyNames([object] $Value) {
    @($Value.PSObject.Properties.Name)
}

function Assert-Object([object] $Value, [string] $Context) {
    if ($null -eq $Value -or $Value -isnot [pscustomobject]) {
        Stop-Redaction "$Context is not an object"
    }
}

function Assert-Array([object] $Value, [string] $Context) {
    if ($null -eq $Value -or $Value -isnot [System.Collections.IList] -or $Value -is [string]) {
        Stop-Redaction "$Context is not an array"
    }
}

function Assert-String([object] $Value, [string] $Context) {
    if ($Value -isnot [string]) {
        Stop-Redaction "$Context is not a string"
    }
}

function Assert-Number([object] $Value, [string] $Context) {
    if ($null -eq $Value -or $Value -is [bool] -or $Value -isnot [ValueType]) {
        Stop-Redaction "$Context is not numeric"
    }
}

function Assert-Boolean([object] $Value, [string] $Context) {
    if ($Value -isnot [bool]) {
        Stop-Redaction "$Context is not boolean"
    }
}

function Assert-PropertySet([object] $Value, [string[]] $Expected, [string] $Context) {
    Assert-Object $Value $Context
    $actual = @(Get-PropertyNames $Value | Sort-Object)
    $expected = @($Expected | Sort-Object)
    if (($actual -join '|') -ne ($expected -join '|')) {
        Stop-Redaction "$Context has an unexpected property set"
    }
}

function Assert-HasProperty([object] $Value, [string] $Name, [string] $Context) {
    Assert-Object $Value $Context
    if ((Get-PropertyNames $Value) -notcontains $Name) {
        Stop-Redaction "$Context is missing required property $Name"
    }
}

function Assert-Marker([object] $Root, [string] $Context) {
    Assert-HasProperty $Root $markerName $Context
    $marker = $Root.$markerName
    Assert-PropertySet $marker @('kind', 'schema_version', 'redacted_fields') "$Context.$markerName"
    Assert-String $marker.kind "$Context.$markerName.kind"
    if ($marker.kind -cne $markerKind) {
        Stop-Redaction "$Context.$markerName.kind is not the expected marker"
    }
    Assert-Number $marker.schema_version "$Context.$markerName.schema_version"
    if ($marker.schema_version -ne $markerVersion) {
        Stop-Redaction "$Context.$markerName.schema_version is unsupported"
    }
    Assert-Array $marker.redacted_fields "$Context.$markerName.redacted_fields"
    $fields = @($marker.redacted_fields)
    if ($fields.Count -ne $markerFields.Count) {
        Stop-Redaction "$Context.$markerName.redacted_fields is incomplete"
    }
    for ($index = 0; $index -lt $markerFields.Count; $index++) {
        Assert-String $fields[$index] "$Context.$markerName.redacted_fields[$index]"
        if ($fields[$index] -cne $markerFields[$index]) {
            Stop-Redaction "$Context.$markerName.redacted_fields is unexpected"
        }
    }
}

function Assert-Game([object] $Game, [string] $Context, [bool] $Public) {
    Assert-Object $Game $Context
    $gameProperties = @(Get-PropertyNames $Game)
    if ($gameProperties -contains 'finite_search_steps') {
        Stop-Redaction "$Context contains an unsupported finite trace"
    }
    Assert-PropertySet $Game @('target', 'outcome', 'path', 'prior_strata', 'posterior_calibration') $Context

    Assert-String $Game.target "$Context.target"
    if ($Game.target.Length -eq 0) {
        Stop-Redaction "$Context.target is empty"
    }
    if ($Public -and $Game.target -cne $redactedWord) {
        Stop-Redaction "$Context.target is not redacted"
    }

    Assert-Array $Game.path "$Context.path"
    for ($pathIndex = 0; $pathIndex -lt $Game.path.Count; $pathIndex++) {
        Assert-String $Game.path[$pathIndex] "$Context.path[$pathIndex]"
        if ($Game.path[$pathIndex].Length -eq 0) {
            Stop-Redaction "$Context.path[$pathIndex] is empty"
        }
        if ($Public -and $Game.path[$pathIndex] -cne $redactedWord) {
            Stop-Redaction "$Context.path[$pathIndex] is not redacted"
        }
    }

    Assert-PropertySet $Game.outcome @('date', 'status', 'guesses') "$Context.outcome"
    Assert-String $Game.outcome.date "$Context.outcome.date"
    Assert-String $Game.outcome.status "$Context.outcome.status"
    if ($null -ne $Game.outcome.guesses) {
        Assert-Number $Game.outcome.guesses "$Context.outcome.guesses"
    }

    if ($null -ne $Game.prior_strata) {
        Assert-PropertySet $Game.prior_strata @('never_used', 'reused', 'historical_only', 'out_of_core') "$Context.prior_strata"
        foreach ($name in @('never_used', 'reused', 'historical_only', 'out_of_core')) {
            Assert-Boolean $Game.prior_strata.$name "$Context.prior_strata.$name"
        }
    }

    Assert-Array $Game.posterior_calibration "$Context.posterior_calibration"
    for ($calibrationIndex = 0; $calibrationIndex -lt $Game.posterior_calibration.Count; $calibrationIndex++) {
        $observation = $Game.posterior_calibration[$calibrationIndex]
        $observationContext = "$Context.posterior_calibration[$calibrationIndex]"
        Assert-PropertySet $observation @('turn', 'score') $observationContext
        Assert-Number $observation.turn "$observationContext.turn"
        if ($null -ne $observation.score) {
            Assert-PropertySet $observation.score @('target_probability', 'log_loss', 'brier') "$observationContext.score"
            foreach ($name in @('target_probability', 'log_loss', 'brier')) {
                Assert-Number $observation.score.$name "$observationContext.score.$name"
            }
        }
    }
}

function Get-GameCollections([object] $Root, [int] $Schema, [string] $Context) {
    Assert-Object $Root $Context
    if ($Schema -eq 4) {
        foreach ($name in @('baseline', 'candidate')) {
            Assert-HasProperty $Root $name $Context
            Assert-HasProperty $Root.$name 'games' "$Context.$name"
            Assert-Array $Root.$name.games "$Context.$name.games"
        }
        return @(
            [pscustomobject]@{ Label = 'baseline'; Games = $Root.baseline.games },
            [pscustomobject]@{ Label = 'candidate'; Games = $Root.candidate.games }
        )
    }

    if ($Schema -eq 7) {
        Assert-HasProperty $Root 'baselines' $Context
        Assert-Array $Root.baselines "$Context.baselines"
        $collections = @()
        for ($baselineIndex = 0; $baselineIndex -lt $Root.baselines.Count; $baselineIndex++) {
            $baselineContext = "$Context.baselines[$baselineIndex]"
            Assert-HasProperty $Root.baselines[$baselineIndex] 'result' $baselineContext
            Assert-HasProperty $Root.baselines[$baselineIndex].result 'games' "$baselineContext.result"
            Assert-Array $Root.baselines[$baselineIndex].result.games "$baselineContext.result.games"
            $collections += [pscustomobject]@{
                Label = "baseline[$baselineIndex]"
                Games = $Root.baselines[$baselineIndex].result.games
            }
        }
        return $collections
    }

    Stop-Redaction "$Context has an unsupported schema"
}

function Assert-RootSchema([object] $Root, [int] $Schema, [string] $Context) {
    Assert-Object $Root $Context
    Assert-HasProperty $Root 'schema_version' $Context
    Assert-Number $Root.schema_version "$Context.schema_version"
    if ($Root.schema_version -ne $Schema) {
        Stop-Redaction "$Context has an unexpected schema version"
    }
    [void](Get-GameCollections $Root $Schema $Context)
}

function Get-JsonRoot([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Stop-Redaction "missing input or public artifact"
    }
    try {
        return (Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json -Depth 100)
    } catch {
        Stop-Redaction "invalid JSON input or public artifact"
    }
}

function Add-RedactionMarker([object] $Root) {
    $marker = [pscustomobject][ordered]@{
        kind = $markerKind
        schema_version = $markerVersion
        redacted_fields = $markerFields
    }
    $Root | Add-Member -NotePropertyName $markerName -NotePropertyValue $marker
}

function Redact-Games([object[]] $Collections) {
    foreach ($collection in $Collections) {
        for ($gameIndex = 0; $gameIndex -lt $collection.Games.Count; $gameIndex++) {
            $game = $collection.Games[$gameIndex]
            $game.target = $redactedWord
            $game.path = @($game.path | ForEach-Object { $redactedWord })
        }
    }
}

function Get-NormalizedJson([object] $Root, [int] $Schema) {
    $clone = Get-JsonRootFromText (ConvertTo-Json -InputObject $Root -Depth 100)
    $collections = @(Get-GameCollections $clone $Schema 'normalized artifact')
    foreach ($collection in $collections) {
        foreach ($game in $collection.Games) {
            $game.target = $redactedWord
            $game.path = @($game.path | ForEach-Object { $redactedWord })
        }
    }
    if ((Get-PropertyNames $clone) -contains $markerName) {
        $clone.PSObject.Properties.Remove($markerName)
    }
    ConvertTo-Json -InputObject $clone -Depth 100 -Compress
}

function Get-JsonRootFromText([string] $Text) {
    try {
        return ($Text | ConvertFrom-Json -Depth 100)
    } catch {
        Stop-Redaction 'failed to normalize JSON artifact'
    }
}

function Get-PathLengths([object[]] $Collections) {
    $lengths = @()
    foreach ($collection in $Collections) {
        foreach ($game in $collection.Games) {
            $lengths += [int]$game.path.Count
        }
    }
    return $lengths
}

function Assert-Equivalent([string] $Expected, [string] $Actual, [string] $Context) {
    if ($Expected -cne $Actual) {
        Stop-Redaction "$Context changed metadata outside the approved redacted fields"
    }
}

function Assert-PublicArtifact([object] $Root, [int] $Schema, [string] $Context) {
    Assert-RootSchema $Root $Schema $Context
    Assert-Marker $Root $Context
    $collections = @(Get-GameCollections $Root $Schema $Context)
    foreach ($collection in $collections) {
        for ($gameIndex = 0; $gameIndex -lt $collection.Games.Count; $gameIndex++) {
            Assert-Game $collection.Games[$gameIndex] "$Context.$($collection.Label).games[$gameIndex]" $true
        }
    }
    return $collections
}

function Write-Utf8Json([string] $Path, [object] $Root) {
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $json = ConvertTo-Json -InputObject $Root -Depth 100 -Compress
    [System.IO.File]::WriteAllText($Path, "$json`n", [System.Text.UTF8Encoding]::new($false))
}

foreach ($spec in $specs) {
    if ($Check) {
        $publicRoot = Get-JsonRoot $spec.Output
        [void](Assert-PublicArtifact $publicRoot $spec.Schema $spec.Name)
        continue
    }

    $sourceRoot = Get-JsonRoot $spec.Input
    Assert-RootSchema $sourceRoot $spec.Schema $spec.Name
    if ((Get-PropertyNames $sourceRoot) -contains $markerName) {
        Stop-Redaction "$spec.Name input already contains a public redaction marker"
    }
    $sourceCollections = @(Get-GameCollections $sourceRoot $spec.Schema $spec.Name)
    foreach ($collection in $sourceCollections) {
        for ($gameIndex = 0; $gameIndex -lt $collection.Games.Count; $gameIndex++) {
            Assert-Game $collection.Games[$gameIndex] "$spec.Name.$($collection.Label).games[$gameIndex]" $false
        }
    }
    $sourceNormalized = Get-NormalizedJson $sourceRoot $spec.Schema
    $sourcePathLengths = @(Get-PathLengths $sourceCollections)

    Redact-Games $sourceCollections
    Add-RedactionMarker $sourceRoot
    Write-Utf8Json $spec.Output $sourceRoot

    $publicRoot = Get-JsonRoot $spec.Output
    $publicCollections = @(Assert-PublicArtifact $publicRoot $spec.Schema $spec.Name)
    $publicPathLengths = @(Get-PathLengths $publicCollections)
    if ($sourcePathLengths.Count -ne $publicPathLengths.Count) {
        Stop-Redaction "$spec.Name changed the number of games"
    }
    for ($index = 0; $index -lt $sourcePathLengths.Count; $index++) {
        if ($sourcePathLengths[$index] -ne $publicPathLengths[$index]) {
            Stop-Redaction "$spec.Name changed a path count"
        }
    }
    Assert-Equivalent $sourceNormalized (Get-NormalizedJson $publicRoot $spec.Schema) $spec.Name
}

if ($Check) {
    Write-Output "Checked $($specs.Count) public evidence artifacts."
} else {
    Write-Output "Generated $($specs.Count) redacted public evidence artifacts."
}
