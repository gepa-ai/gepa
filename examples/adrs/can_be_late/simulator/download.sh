# rsync -Pavz search-ddl:~/sky_workdir/results/real/ddl=search+task=76+overhead=0.30/* ./results/real/ddl\=search+task\=76+overhead\=0.30/
# rsync -Pavz all-slices:~/sky_workdir/results/real/ddl=search+task=76+overhead=0.50/* ./results/real/ddl\=search+task\=76+overhead\=0.50/
# python scripts/download.py ddl-52-01-05:~/sky_workdir/results/real/ddl=search+task=48* ./results/real/


python scripts/download.py zhwu1:~/sky_workdir/results/real/ddl=search+task=48* ./results/real/
python scripts/download.py zhwu1:~/sky_workdir/results/real_complete/ddl=search+task=48* ./results/real_complete/

python scripts/download.py zhwu2:~/sky_workdir/results/real/ddl=search+task=48* ./results/real/
python scripts/download.py zhwu2:~/sky_workdir/results/real_complete/ddl=search+task=48* ./results/real_complete/

python scripts/download.py zhwu3:~/sky_workdir/results/real/2023-02-15/ddl=search+task=48* ./results/real/2023-02-15/

# rsync -Pavz zhwu1:~/sky_workdir/exp/real/ddl=60+task=48* ./exp/real/
# python scripts/download.py zhwu1:~/sky_workdir/results/real/ddl=60+task=48* ./results/real/
# python scripts/download.py zhwu1:~/sky_workdir/results/real/real_preemption/* ./results/real/real_preemption/


mkdir -p ./results-plot
python scripts/download.py zhwu1:~/sky_workdir/results-plot/real/ddl=search* ./results-plot/real/
rsync -Pavz zhwu1:~/sky_workdir/exp-for-plot/ ./exp-for-plot/
python scripts/download.py zhwu1:~/sky_workdir/results/real/ddl=search* ./results/real/
