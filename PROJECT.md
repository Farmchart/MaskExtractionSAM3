# TODO
+ Separation of area segmentation and object segmentation
+ Preset groups for common cases (e.g. plants, people, tractors, etc.)
+ Adjustable inference res for different objects, then downscale for priority subtraction

# Main workflow (so far)
Workflow:
1. Generate masks
    python extract_masks_sam3.py --image ./images/ \ <br>
        --group plants "crop plants" "leaves" \ <br>
        --group persons "person" "human"
<br>
2. Run COLMAP
    colmap feature_extractor \ <br>
        --database_path scene/database.db \ <br>
        --image_path scene/images
<br>
3. Train a splat per group in LichtFeld
    ./LichtFeld-Studio -d scene/ -o output/plants/ --mask-path masks/plants/ <br>
    ./LichtFeld-Studio -d scene/ -o output/persons/ --mask-path masks/persons/ <br>
<br>
4. Overlay the PLY files on the base splat in LichtFeld
