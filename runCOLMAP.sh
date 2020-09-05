DATA_PATH=./data

CUDA_VISIBLE_DEVICES=6 colmap automatic_reconstructor \
    --workspace_path ${DATA_PATH} \
    --image_path ${DATA_PATH}/images \
    --camera_model SIMPLE_PINHOLE \
    --dense 0