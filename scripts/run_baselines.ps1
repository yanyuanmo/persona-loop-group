$ErrorActionPreference = 'Stop'

$agents = @('persona_loop', 'continuous', 'periodic_remind', 'rag', 'sliding_window', 'ppa')

foreach ($agent in $agents) {
    python run_experiment.py agent=$agent experiment.run_name="baseline_$agent"
}
