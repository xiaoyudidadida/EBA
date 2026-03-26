#!/bin/bash

# ============================================================
# SGF-RRA-MEB 消融实验脚本
# ============================================================
# 用法:
#   bash run_IEMOCAP.sh
#   bash run_IEMOCAP.sh --rra     (只跑 RRA 消融)
#   bash run_IEMOCAP.sh --meb     (只跑 MEB 消融)
#   bash run_IEMOCAP.sh --full    (跑完整模型)
#   bash run_IEMOCAP.sh --all     (跑全部 4 组消融)
# ============================================================

# ---- 基础参数配置（所有实验共享） ----
BASE_ARGS="--base-model GRU \
    --dropout 0.4 \
    --lr 0.0001 \
    --batch-size 16 \
    --graph_type hyper \
    --epochs 80 \
    --graph_construct direct \
    --multi_modal \
    --mm_fusion_mthd concat_DHT \
    --modals avl \
    --Dataset IEMOCAP \
    --norm BN \
    --proto_k 1 \
    --class-weight"

# ---- 随机种子 ----
# 固定种子保证可复现性: IEMOCAP=1746, MELD=67137
# 如需多次随机实验，请通过外部脚本动态传入不同 seed
SEED_ARGS="--seed 1746"

# ---- 辅助函数 ----
run_exp() {
    local name="$1"
    shift
    echo ""
    echo "========================================"
    echo ">>> 开始运行: $name"
    echo "========================================"
    python -u train_IEMOCAP.py $BASE_ARGS $SEED_ARGS "$@"
}

# ---- 解析参数，决定运行哪些实验 ----
MODE="${1:-all}"

case "$MODE" in
    --rra)
        echo ">>> 模式: 仅 RRA 消融"
        run_exp "SGF + RRA" --use_rra
        ;;

    --meb)
        echo ">>> 模式: 仅 MEB 消融"
        run_exp "SGF + MEB" --use_meb
        ;;

    --full)
        echo ">>> 模式: 完整模型 (SGF + RRA + MEB)"
        run_exp "SGF + RRA + MEB" --use_rra --use_meb
        ;;

    --all|*)
        echo ">>> 模式: 全部 4 组消融实验"
        echo ""
        echo "========================================"
        echo ">>> [1/4] 基线模型 (SGF)"
        echo "========================================"
        python -u train_IEMOCAP.py $BASE_ARGS $SEED_ARGS

        echo ""
        echo "========================================"
        echo ">>> [2/4] SGF + RRA"
        echo "========================================"
        python -u train_IEMOCAP.py $BASE_ARGS $SEED_ARGS --use_rra

        echo ""
        echo "========================================"
        echo ">>> [3/4] SGF + MEB"
        echo "========================================"
        python -u train_IEMOCAP.py $BASE_ARGS $SEED_ARGS --use_meb

        echo ""
        echo "========================================"
        echo ">>> [4/4] SGF + RRA + MEB (完整模型)"
        echo "========================================"
        python -u train_IEMOCAP.py $BASE_ARGS $SEED_ARGS --use_rra --use_meb

        echo ""
        echo "========================================"
        echo ">>> 全部实验完成!"
        echo "========================================"
        ;;
esac
