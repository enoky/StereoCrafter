<div dir="auto"><section><div dir="auto"><div dir="auto"><div class="markdown-heading" dir="auto"><h1 tabindex="-1" dir="auto" class="heading-element">Depth Splatting Batch Processor</h1><a id="user-content-depth-splatting-batch-processor" class="anchor" aria-label="Permalink: Depth Splatting Batch Processor" href="#depth-splatting-batch-processor"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-depth-splatting-batch-processor" aria-label="Permalink: Depth Splatting Batch Processor" href="#depth-splatting-batch-processor"></a></div>
<p dir="auto">A Python script for batch processing videos to generate depth maps and perform depth-based splatting. It leverages the DepthCrafter pipeline for depth inference or utilizes pre-rendered depth maps to create enhanced splatted videos with stereo projections.</p>
<div dir="auto"><div class="markdown-heading" dir="auto"><h2 tabindex="-1" dir="auto" class="heading-element">Features</h2><a id="user-content-features" class="anchor" aria-label="Permalink: Features" href="#features"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-features" aria-label="Permalink: Features" href="#features"></a></div>
<ul dir="auto">
<li><strong>Batch Processing</strong>: Handle multiple videos in a single run.</li>
<li><strong>Depth Inference</strong>: Generate depth maps using the DepthCrafter pipeline.</li>
<li><strong>Pre-rendered Depth Support</strong>: Use existing depth maps to skip inference.</li>
<li><strong>Stereo Projection</strong>: Apply forward warping for depth-based splatting.</li>
<li><strong>Configurable Parameters</strong>: Customize processing settings like resolution, FPS, and batch size.</li>
</ul>
<div dir="auto"><div class="markdown-heading" dir="auto"><h3 tabindex="-1" dir="auto" class="heading-element">Command Syntax</h3><a id="user-content-command-syntax" class="anchor" aria-label="Permalink: Command Syntax" href="#command-syntax"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-command-syntax" aria-label="Permalink: Command Syntax" href="#command-syntax"></a></div>
<div dir="auto"><pre>python depth_splatting.py \
  --input_source_clips <span>&lt;</span>path_to_input_videos<span>&gt;</span> \
  --input_depth_maps <span>&lt;</span>path_to_pre_rendered_depth_maps<span>&gt;</span> \
  --output_splatted <span>&lt;</span>path_to_output_videos<span>&gt;</span> \
  --unet_path <span>&lt;</span>path_to_unet_model<span>&gt;</span> \
  --pre_trained_path <span>&lt;</span>path_to_depthcrafter_model<span>&gt;</span> \
  --max_disp <span>&lt;</span>max_disparity<span>&gt;</span> \
  --process_length <span>&lt;</span>number_of_frames<span>&gt;</span> \
  --batch_size <span>&lt;</span>batch_size<span>&gt;</span></pre></div>
<div dir="auto"><div class="markdown-heading" dir="auto"><h3 tabindex="-1" dir="auto" class="heading-element">Parameters</h3><a id="user-content-parameters" class="anchor" aria-label="Permalink: Parameters" href="#parameters"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-parameters" aria-label="Permalink: Parameters" href="#parameters"></a></div>
<ul dir="auto">
<li>
<p dir="auto"><code>input_source_clips</code> (str): Path to the folder containing input source videos.<br>
<strong>Default</strong>: <code>./input_source_clips</code></p>
</li>
<li>
<p dir="auto"><code>input_depth_maps</code> (str): Path to the folder containing pre-rendered depth maps.<br>
<strong>Default</strong>: <code>./input_depth_maps</code></p>
</li>
<li>
<p dir="auto"><code>output_splatted</code> (str): Path to the folder where output splatted videos will be saved.<br>
<strong>Default</strong>: <code>./output_splatted</code></p>
</li>
<li>
<p dir="auto"><code>unet_path</code> (str): Path to the pre-trained UNet model. Required if not using pre-rendered depth maps.<br>
<strong>Default</strong>: <code>./weights/DepthCrafter</code></p>
</li>
<li>
<p dir="auto"><code>pre_trained_path</code> (str): Path to the pre-trained DepthCrafter model. Required if not using pre-rendered depth maps.<br>
<strong>Default</strong>: <code>./weights/stable-video-diffusion-img2vid-xt-1-1</code></p>
</li>
<li>
<p dir="auto"><code>max_disp</code> (float): Maximum disparity for splatting.<br>
<strong>Default</strong>: <code>20.0</code></p>
</li>
<li>
<p dir="auto"><code>process_length</code> (int): Number of frames to process per video. <code>-1</code> for all frames.<br>
<strong>Default</strong>: <code>-1</code></p>
</li>
<li>
<p dir="auto"><code>batch_size</code> (int): Batch size for processing to manage GPU memory.<br>
<strong>Default</strong>: <code>10</code></p>
</li>
</ul>
<div dir="auto"><div class="markdown-heading" dir="auto"><h3 tabindex="-1" dir="auto" class="heading-element">Examples</h3><a id="user-content-examples" class="anchor" aria-label="Permalink: Examples" href="#examples"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-examples" aria-label="Permalink: Examples" href="#examples"></a></div>
<ol dir="auto">
<li>
<p dir="auto"><strong>Process Videos with Depth Inference</strong></p>
<div dir="auto"><pre>python depth_splatting.py \
  --input_source_clips ./input_videos \
  --input_depth_maps ./input_depth_maps \
  --output_splatted ./output \
  --unet_path ./weights/DepthCrafter \
  --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
  --max_disp 20.0 \
  --process_length -1 \
  --batch_size 10</pre></div>
</li>
<li>
<p dir="auto"><strong>Process Videos Using Pre-rendered Depth Maps Only</strong></p>
<div dir="auto"><pre>python depth_splatting.py \
  --input_source_clips ./input_videos \
  --input_depth_maps ./pre_rendered_depth \
  --output_splatted ./output \
  --max_disp 20.0 \
  --batch_size 10</pre></div>
<p dir="auto"><em>Omit <code>unet_path</code> and <code>pre_trained_path</code> if only using pre-rendered depth maps.</em></p>
</li>
</ol>
<div dir="auto"><div class="markdown-heading" dir="auto"><h2 tabindex="-1" dir="auto" class="heading-element">Notes</h2><a id="user-content-notes" class="anchor" aria-label="Permalink: Notes" href="#notes"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-notes" aria-label="Permalink: Notes" href="#notes"></a></div>
<ul dir="auto">
<li>
<p dir="auto"><strong>Pre-rendered Depth Maps</strong>: Ensure that pre-rendered depth maps are named consistently with their corresponding source videos (e.g., <code>video1.mp4</code> and <code>video1_depth.mp4</code>).</p>
</li>
<li>
<p dir="auto"><strong>GPU Usage</strong>: Depth inference and splatting are GPU-intensive. Ensure sufficient GPU memory is available, especially when processing large batches.</p>
</li>
<li>
<p dir="auto"><strong>Output Structure</strong>: Splatted videos will be saved in the specified <code>output_splatted</code> directory with the suffix <code>_splatted.mp4</code>.</p>
</li>
</ul>
<div dir="auto"><div class="markdown-heading" dir="auto"><h2 tabindex="-1" dir="auto" class="heading-element">Troubleshooting</h2><a id="user-content-troubleshooting" class="anchor" aria-label="Permalink: Troubleshooting" href="#troubleshooting"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-troubleshooting" aria-label="Permalink: Troubleshooting" href="#troubleshooting"></a></div>
<ul dir="auto">
<li>
<p dir="auto"><strong>Missing Models</strong>: Ensure that the UNet and DepthCrafter models are correctly placed in the specified directories.</p>
</li>
<li>
<p dir="auto"><strong>Dependencies Issues</strong>: Verify that all required Python libraries are installed. Use <code>pip install -r requirements.txt</code> to install missing packages.</p>
</li>
<li>
<p dir="auto"><strong>Performance</strong>: Adjust <code>batch_size</code> and <code>process_length</code> based on your hardware capabilities to optimize performance and memory usage.</p>
</li>
</ul>
</div></section></div>
