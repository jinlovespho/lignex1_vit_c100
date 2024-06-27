#!/bin/bash

# 파라미터 조합
batch_sizes=(64 128 256)
learning_rates=(1e-3 5e-4)
weight_decays=(1e-4 1e-5)
warm_ups=(5 10)
dropouts=(0.2 0.1)

# 동시에 실행할 최대 실험 수
MAX_JOBS=2

# 각 조합에 대해 실험 실행
for batch_size in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for weight_decay in "${weight_decays[@]}"; do
            for dropout in "${dropouts[@]}"; do
                for warm_ups in "${warm_ups[@]}"; do
                    # 백그라운드 작업 수 확인 및 대기
                    while [ $(jobs | wc -l) -ge $MAX_JOBS ]; do
                        sleep 1
                    done

                    # 실험 실행
                    echo "Running experiment with batch_size=$batch_size, lr=$lr, weight_decay=$weight_decay, dropout=$dropout, warm_ups = $warm_ups"
                    python main.py --batch-size $batch_size --lr $lr --weight-decay $weight_decay --dropout $dropout --warmup-epoch $warm_ups &
                done
            done
        done
    done
done

# 모든 백그라운드 작업이 완료될 때까지 대기
wait
echo "All experiments completed."