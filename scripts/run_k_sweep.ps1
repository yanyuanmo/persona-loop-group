$ErrorActionPreference = 'Stop'

$kValues = @(2, 4, 6, 8)
foreach ($k in $kValues) {
    python run_experiment.py experiment.run_name="k_$k" experiment.k=$k
}
