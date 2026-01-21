wandb offline
export LOG_LEVEL=DEBUG

TRACE_PATH=data/real/ping_based/random_start_time/us-west-2a_v100_1
# TRACE_PATH=data/two_gamma/gap_600-alive_5.0_0.1-wait_5.0_0.1/gap_600-alive_5.0_0.1-wait_5.0_0.1_random
# TRACE_PATH=data/two_exp/gap_600-alive_0.02-wait_0.02/gap_600-alive_0.02-wait_0.02_random
# TRACE_PATH=data/two_exp/gap_600-alive_0.04082965866405357-wait_0.1559405943681992/exp_gap_600-alive_0.04082965866405357-wait_0.1559405943681992_random
# TRACE_PATH=data/two_exp/gap_600-preemption_rate_0.9/gap_600-preemption_rate_0.9_random
# TRACE_PATH=data/real/ping_based/random_start_time/us-west-2a_k80_1
# TRACE_PATH=data/two_gamma/gap_600-alive_0.1919_0.008700000000002435-wait_0.358_0.06579999999981707/random_start_time
# TRACE_PATH=data/two_exp/gap_60-real_mean/gap_60-real_mean_random
# TRACE_PATH=data/two_exp/gap_600-real_mean/exp_gap_600-real_mean

RESTART_OVERHEAD_HOURS=0
basename=$(basename $TRACE_PATH)$RESTART_OVERHEAD_HOURS


# python ./main.py --strategy=on_demand \
#                 --env trace \
#                 --trace-file $TRACE_PATH \
#                 --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                 --deadline-hours=1000 \
#                 --output-dir exp-new-sliced \
#                 --task-duration-hours=48 | tee ./$basename-on-demand.out 2>&1 &

# python ./main.py --strategy=only_spot \
#                 --env trace \
#                 --trace-file $TRACE_PATH \
#                 --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                 --deadline-hours=1000 \
#                 --output-dir exp-new-sliced \
#                 --task-duration-hours=48 | tee ./$basename-only_spot.out 2>&1 &

# python ./main.py --strategy=strawman \
#                 --env trace \
#                 --trace-file $TRACE_PATH \
#                 --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                 --deadline-hours=52 \
#                 --output-dir exp-new-sliced \
#                 --task-duration-hours=48 | tee ./$basename-strawman.out 2>&1

# wait

# for i in 1 2 4 8 16; do
#     python ./main.py --strategy=time_sliced \
#                     --slice-interval-hours=$i \
#                     --use-avg-gain \
#                     --env trace \
#                     --trace-file $TRACE_PATH \
#                     --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                     --deadline-hours=52 \
#                     --output-dir exp-new-sliced \
#                     --task-duration-hours=48 | tee ./$basename-slice-avg-$i.out 2>&1 &

#     python ./main.py --strategy=time_sliced \
#                     --slice-interval-hours=$i \
#                     --env trace \
#                     --trace-file $TRACE_PATH \
#                     --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                     --deadline-hours=52 \
#                     --output-dir exp-new-sliced \
#                     --task-duration-hours=48 | tee ./$basename-slice-$i.out 2>&1 &

#     python ./main.py --strategy=loose_time_sliced \
#                     --slice-interval-hours=$i \
#                     --use-avg-gain \
#                     --env trace \
#                     --trace-file $TRACE_PATH \
#                     --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                     --deadline-hours=52 \
#                     --output-dir exp-new-sliced \
#                     --task-duration-hours=48 | tee ./$basename-slice-avg-$i.out 2>&1 &

#     python ./main.py --strategy=loose_time_sliced \
#                     --slice-interval-hours=$i \
#                     --env trace \
#                     --trace-file $TRACE_PATH \
#                     --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                     --deadline-hours=52 \
#                     --output-dir exp-new-sliced \
#                     --task-duration-hours=48 | tee ./$basename-slice-$i.out 2>&1 &
#     wait
# done

# for i in 1 2 4 8 16; do
#     for slice_slack in 1 2 4 6 8; do
#         python ./main.py --strategy=loose_time_sliced \
#                         --slice-interval-hours=$i \
#                         --use-avg-gain \
#                         --env trace \
#                         --trace-file $TRACE_PATH \
#                         --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                         --deadline-hours=52 \
#                         --task-duration-hours=48 \
#                         --output-dir exp-new-sliced \
#                         --max-slice-slacks=$slice_slack &
#     done
#     wait

#     for slice_slack in 1 2 4 6 8; do
#         python ./main.py --strategy=loose_time_sliced \
#                         --slice-interval-hours=$i \
#                         --env trace \
#                         --trace-file $TRACE_PATH \
#                         --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                         --deadline-hours=52 \
#                         --task-duration-hours=48 \
#                         --output-dir exp-new-sliced \
#                         --max-slice-slacks=$slice_slack &
#     done
#     wait

#     for total_slack in 8 12 16 20 24; do
#         python ./main.py --strategy=loose_time_sliced \
#                         --slice-interval-hours=$i \
#                         --use-avg-gain \
#                         --env trace \
#                         --trace-file $TRACE_PATH \
#                         --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                         --deadline-hours=52 \
#                         --task-duration-hours=48 \
#                         --output-dir exp-new-sliced \
#                         --max-total-slacks=$total_slack &
#     done
#     wait

#     for total_slack in 8 12 16 20 24; do
#         python ./main.py --strategy=loose_time_sliced \
#                         --slice-interval-hours=$i \
#                         --env trace \
#                         --trace-file $TRACE_PATH \
#                         --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                         --deadline-hours=52 \
#                         --task-duration-hours=48 \
#                         --output-dir exp-new-sliced \
#                         --max-total-slacks=$total_slack &
#     done
#     wait
# done

# Group time sliced
# for i in 1 2 4 8 16; do
#     for j in 1 2 4 8 16; do
#         python ./main.py --strategy=group_time_sliced \
#                         --slice-interval-hours-groups $i $j \
#                         --use-avg-gain \
#                         --env trace \
#                         --trace-file $TRACE_PATH \
#                         --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                         --deadline-hours=52 \
#                         --output-dir exp-new-sliced \
#                         --task-duration-hours=48 &
#     done
#     wait

#     for j in 1 2 4 8 16; do
#         python ./main.py --strategy=group_time_sliced \
#                         --slice-interval-hours-groups $i $j \
#                         --env trace \
#                         --trace-file $TRACE_PATH \
#                         --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                         --deadline-hours=52 \
#                         --output-dir exp-new-sliced \
#                         --task-duration-hours=48 &
#     done
#     wait
# done

# python ./main.py --strategy=random_time_sliced \
#                 --slice-interval-hours-choices 2 4 8 \
#                 --use-avg-gain \
#                 --env trace \
#                 --trace-file $TRACE_PATH \
#                 --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                 --deadline-hours=52 \
#                 --output-dir exp-new-sliced \
#                 --task-duration-hours=48 | tee ./$basename-random-slice-avg-$i.out 2>&1

# python ./main.py --strategy=random_time_sliced \
#                 --slice-interval-hours-choices 2 4 8 \
#                 --env trace \
#                 --trace-file $TRACE_PATH \
#                 --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                 --deadline-hours=52 \
#                 --output-dir exp-new-sliced \
#                 --task-duration-hours=48 | tee ./$basename-random-slice-$i.out 2>&1

# python ./main.py --strategy=ideal_ilp_overhead \
#                 --env trace \
#                 --trace-file $TRACE_PATH \
#                 --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
#                 --deadline-hours=52 \
#                 --output-dir exp-new-sliced \
#                 --task-duration-hours=48

python ./main.py --strategy=ideal_no_overhead \
                --env trace \
                --trace-file $TRACE_PATH \
                --restart-overhead-hours=$RESTART_OVERHEAD_HOURS \
                --deadline-hours=52 \
                --output-dir exp-new-sliced \
                --task-duration-hours=48


                # --slice-interval-hours=8 \
                # --use-avg-gain \
                # --trace-file data/poisson/gap_600-hourly_rate_$rate \
                # --trace-file data/real/ping_based/us-west-2a_v100_1.txt \
                # --env-start-hours=72 \

                # --restart-overhead-hours=0.5 \
                # --trace-file data/poisson/gap_1200-hourly_rate_0.1/1.json \
                # --trace-file data/real/analysis/1k80us-east-1c.txt \
                # --slice-interval-hours=8 \
                # --use-avg-gain \
