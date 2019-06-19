#!/bin/bash
#$ -l h_rt=8:00:00  #time needed
#$ -pe smp 4 #number of cores
#$ -l rmem=4G #number of memery
#$ -o Q2_2b.txt #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M zli132@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory
#$ -P rse-com6012
#$ -q rse-com6012.q

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit ../Code/Regression_Linear.py
