param(
    [ValidateSet("prepare", "pretrain", "sft", "lora", "dpo", "grpo", "agent", "eval", "benchmark", "tool_eval", "serve", "all")]
    [string]$Stage = "prepare",
    [string]$Device = "cuda:0",
    [string]$Weight = "full_sft",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Runner = Join-Path $Root "scripts\run_dronemind.py"

$ArgsList = @($Runner, $Stage, "--device", $Device, "--weight", $Weight)
if ($DryRun) {
    $ArgsList += "--dry_run"
}

python @ArgsList
