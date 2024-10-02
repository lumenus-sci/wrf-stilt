for alt in 50 150 250 500 750 1000 1500 2000 3000 4000 5000 7500 10000; do
    sbatch sbatch_sample_wrfout.sh d01 ${alt}
done
