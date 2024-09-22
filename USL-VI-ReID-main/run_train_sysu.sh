CUDA_VISIBLE_DEVICES=2,3 \
python "/home/shaorui/USL-VI-ReID-main/train_sysu.py" -b 128 -a agw -d  sysu_all \
--num-instances 16 \
--data-dir "/home/shaorui/USL-VI-ReID-main/data/SYSU-MM01/" \
--logs-dir "/home/shaorui/USL-VI-ReID-main/data/logs/" \