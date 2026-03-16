# TODO
+ Separation of area segmentation and object segmentation
+ Preset groups for common cases (e.g. plants, people, tractors, etc.)
+ Adjustable inference res for different objects, then downscale for priority subtraction
+ Continuous tracking of objects between frames for consistent masking

# Main workflow (so far)

Workflow:

1. Generate masks
```bash
python extract_masks_sam3.py --image ./images/ \
    --group plants "crop plants" "leaves" \
    --group persons "person" "human"
```

2. Run COLMAP
```bash
colmap feature_extractor \
    --database_path scene/database.db \
    --image_path scene/images
```

3. Train a splat per group in LichtFeld
```bash
./LichtFeld-Studio -d scene/ -o output/plants/ --mask-path masks/plants/
./LichtFeld-Studio -d scene/ -o output/persons/ --mask-path masks/persons/
```

4. Overlay the PLY files on the base splat in LichtFeld
