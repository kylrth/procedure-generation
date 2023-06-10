#!/bin/bash
# remember to run slurm_env.sh once first
#
#SBATCH --time=0-01:20:00
#SBATCH --mem=32000
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --array=1-10

set -e

# environment
module load python/3.10 scipy-stack cuda cudnn gcc/9.3.0 arrow/11.0.0
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --quiet --upgrade pip
pip install --no-index --quiet datasets torch sentence_transformers

# bring datasets cache to local, for performance with transformations
# no need to bring the actual dataset, because it's a single file and will be read only once
cp -r ~/scratch/datasets_cache $SLURM_TMPDIR/datasets_cache

beforehand=$(date +%s)
python embed.py train \
    --data-dir ~/scratch/recipenlg/ \
    --model-cache ~/scratch/torch_cache \
    --datasets-cache $SLURM_TMPDIR/datasets_cache \
    --chunk $SLURM_ARRAY_TASK_ID --chunks 10 \
    --output-root ~/scratch/embeddings

if [[ $(find $SLURM_TMPDIR/datasets_cache -type f -newermt @$beforehand) ]]; then
    echo "datasets cache modified; copying back to scratch"
    rm -rf ~/scratch/datasets_cache_old
    mv ~/scratch/datasets_cache ~/scratch/datasets_cache_old
    cp -r $SLURM_TMPDIR/datasets_cache ~/scratch/datasets_cache
else
    echo "datasets cache not modified"
fi
