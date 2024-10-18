!/bin/bash
SBATCH --job-name=Test_models   # create a short name for your job
SBATCH --output=./outputs/out.%J	# job stdout file
SBATCH --error=./outputs/err.%J
SBATCH --nodes=1                # node count
SBATCH --ntasks=12               # total number of tasks across all nodes, Numero de processadores,cores
SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks),
SBATCH --mem=4G                 # total memory per node (4 GB per cpu-core is default)
SBATCH --time=96:00:00          # total run time limit (HH:MM:SS)
SBATCH --mail-user=alves_lucasoliveira@usp.br
SBATCH --mail-type=END

module purge
module load Anaconda/3-2022.05

python testing_models.py
