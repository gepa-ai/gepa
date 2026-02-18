#!/usr/bin/env fish

export LOG_LEVEL=INFO

set REGION1 data/real/ping_based/random_start_time/us-west-2a_v100_1
set REGION2 data/real/ping_based/random_start_time/us-west-2b_v100_1
set OUTPUT_DIR exp-multi-region-time-sliced

mkdir -p $OUTPUT_DIR

echo "Running single-region time_sliced strategy (baseline)..."
python ./main.py --strategy=time_sliced \
                --slice-interval-hours=6 \
                --env=trace \
                --trace-file=$REGION1 \
                --restart-overhead-hours=0.2 \
                --deadline-hours=52 \
                --output-dir=$OUTPUT_DIR \
                --task-duration-hours=48

echo "Running multi-region time_sliced strategy with multi-region awareness..."
python ./main.py --strategy=multi_region_time_sliced \
                --slice-interval-hours=6 \
                --env=multi_trace \
                --trace-files $REGION1 $REGION2 \
                --restart-overhead-hours=0.2 \
                --deadline-hours=52 \
                --output-dir=$OUTPUT_DIR \
                --task-duration-hours=48 \
                --region-switch-overhead-hours=0.1 \

echo "Done! Results saved to $OUTPUT_DIR" 