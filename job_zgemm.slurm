#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey0001-gpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive

module load craype-accel-amd-gfx90a
module load rocm
module load slate-amd-gfx90a/2024.10.29 


export OMP_NUM_THREADS=1

export MPICH_GPU_SUPPORT_ENABLED=1
export SLATE_GPU_AWARE_MPI=1

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu


CPU_BIND=$(generate_CPU_BIND.sh mask_cpu) #Note the use of mask_cpu and not map
echo "CPU_BIND=${CPU_BIND}"

#CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
#CPU_BIND="${CPU_BIND},fe0000,fe000000"
#CPU_BIND="${CPU_BIND},fe,fe00"
#CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"


rm *.o *.mod demo_zmatmat
./zcompile

#srun -u  -n 8 -c 1 ./demo_zmatvec 
srun -u  -n 8 -c 8 --cpu-bind=${CPU_BIND} ./select_gpu ./demo_zmatmat

ftn demo_zmatmat_cpu.f90 -o demo_zmatmat_cpu

echo "Now running the CPU version"

srun -n 1 -c 1 ./demo_zmatmat_cpu
