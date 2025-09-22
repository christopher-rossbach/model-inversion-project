DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch.tinygpu $DIR/job.slurm /home/woody/iwi1/iwi1106h/data/imagenet/train/n02086646/n02086646_567.JPEG
sbatch.tinygpu $DIR/job.slurm Images/14158.jpg
sbatch.tinygpu $DIR/job.slurm Images/14042.jpg