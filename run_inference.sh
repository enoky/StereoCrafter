# python depth_splatting_inference.py \
#     --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1\
#     --unet_path ./weights/DepthCrafter \
#     --input_video_path ./source_video/camel.mp4 \
#     --output_video_path ./outputs/camel_splatting_results.mp4

# python inpainting_inference.py \
#     --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
#     --unet_path ./weights/StereoCrafter \
#     --input_video_path ./outputs/camel_splatting_results.mp4 \
#     --save_dir ./outputs \
#     --tile_num 2
