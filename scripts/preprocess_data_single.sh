export DATASET=$1
export MESH_NAME=$2
export OBJECT=$3
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="$MESH_NAME/render/images"
export OUTPUT_DIR="textual_inversion/models/$MESH_NAME"

# normalize mesh
python utils/normalize_mesh.py -f "$MESH_NAME" --dataset_type "$DATASET" --norm_with_smplx

# render rgb for segmentation
python segmentation/render.py -f "$MESH_NAME"

# segmentation
python segmentation/grounded_sam_demo.py \
  --config segmentation/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint segmentation/groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint segmentation/sam_hq_vit_h.pth \
  --use_sam_hq \
  --input_image $MESH_NAME \
  --output_dir "$MESH_NAME/render/segms" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "$OBJECT" \
  --device "cuda"

# vote segmentation in 3D
python segmentation/segm_3d.py -f "$MESH_NAME" --dataset_type "$DATASET"