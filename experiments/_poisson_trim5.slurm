#!/bin/bash
#SBATCH --job-name=poisson5
#SBATCH --output=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/poisson_trim5_%j.log
#SBATCH --error=/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/logs/poisson_trim5_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu_devel
#SBATCH --time=01:00:00

cd /home/glenain/Sphere-traced-1-Lipschitz-MLPs-

WS=${WS:-outputs/colmap_barn_halfres}
TRIM=${TRIM:-5}

echo "[$(date)] poisson_mesher trim=$TRIM"
colmap poisson_mesher \
    --input_path  $WS/fused.ply \
    --output_path $WS/poisson_trim${TRIM}.ply \
    --PoissonMeshing.trim $TRIM
echo "[$(date)] done"
ls -lh $WS/poisson_trim${TRIM}.ply
