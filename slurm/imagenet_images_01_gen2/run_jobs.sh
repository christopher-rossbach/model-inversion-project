DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch.tinygpu $DIR/job.slurm /home/woody/iwi1/iwi1106h/data/imagenet/train/n01440764/n01440764_44.JPEG
sbatch.tinygpu $DIR/job.slurm /home/woody/iwi1/iwi1106h/data/imagenet/train/n02086646/n02086646_567.JPEG $DIR/job.slurm /home/woody/iwi1/iwi1106h/data/imagenet/train/n04125021/n04125021_4187.JPEG