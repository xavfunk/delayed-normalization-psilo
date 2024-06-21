#!/bin/bash
#$ -N optseq2
#$ -cwd
#$ -j y                                           # Job error stream is merged with output stream
#$ -q long.q                                 # Queue to use
#$ -u funk                            # Username
#$ -V
#$ -e /data1/projects/dumoulinlab/Lab_members/Xaver/temp/optseq/optseq2$JOB_ID-err.log
#$ -o /data1/projects/dumoulinlab/Lab_members/Xaver/temp/optseq/optseq2$JOB_ID-out.log
#$ -M funk@spinozacentre.nl


optseq2 --ntp 200 --tr 1.6 --psdwin 0 12.8 1.6 --ev dur_0 4.8 3 --ev dur_2 4.8 3 --ev dur_4 4.8 3 --ev dur_8 4.8 3 --ev dur_16 4.8 3 --ev dur_32 4.8 3 --ev dur_64 4.8 3 --ev isi_2 4.8 3 --ev isi_4 4.8 3 --ev isi_8 4.8 3 --ev isi_16 4.8 3 --ev isi_32 4.8 3 --ev isi_64 4.8 3 --nkeep 15 --o tr-200_min-3_tsearch-12 --tsearch 12
