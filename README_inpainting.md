<div dir="auto"><section><div dir="auto"><div dir="auto"><div class="markdown-heading" dir="auto"><h1 tabindex="-1" dir="auto" class="heading-element">Video Inpainting with Stable Video Diffusion</h1><a id="user-content-video-inpainting-with-stable-video-diffusion" class="anchor" aria-label="Permalink: Video Inpainting with Stable Video Diffusion" href="#video-inpainting-with-stable-video-diffusion"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-video-inpainting-with-stable-video-diffusion" aria-label="Permalink: Video Inpainting with Stable Video Diffusion" href="#video-inpainting-with-stable-video-diffusion"></a></div>
<p dir="auto">This script processes videos for inpainting tasks using a stable video diffusion pipeline. It supports batch processing of videos, spatial tiling for large frames, and GPU acceleration for fast processing.</p>
<div dir="auto"><div class="markdown-heading" dir="auto"><h2 tabindex="-1" dir="auto" class="heading-element">Features</h2><a id="user-content-features" class="anchor" aria-label="Permalink: Features" href="#features"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-features" aria-label="Permalink: Features" href="#features"></a></div>
<ul dir="auto">
<li><strong>Batch Processing</strong>: Handle multiple videos in a single run.</li>
<li><strong>Spatial Tiling</strong>: Processes videos in tiles to handle high-resolution frames efficiently.</li>
<li><strong>Video Inpainting</strong>: Fills in missing parts of a video using a pre-trained inpainting pipeline.</li>
<li><strong>Configurable Parameters</strong>: Adjust chunk size, overlap, and tiling for optimal performance.</li>
</ul>
<div dir="auto"><div class="markdown-heading" dir="auto"><h3 tabindex="-1" dir="auto" class="heading-element">Command Syntax</h3><a id="user-content-command-syntax" class="anchor" aria-label="Permalink: Command Syntax" href="#command-syntax"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-command-syntax" aria-label="Permalink: Command Syntax" href="#command-syntax"></a></div>
<div dir="auto"><pre>python inpainting.py \
  --pre_trained_path <span>&lt;</span>path_to_pipeline_weights<span>&gt;</span> \
  --unet_path <span>&lt;</span>path_to_unet_weights<span>&gt;</span> \
  --input_folder <span>&lt;</span>path_to_input_videos<span>&gt;</span> \
  --output_folder <span>&lt;</span>path_to_output_videos<span>&gt;</span> \
  --frames_chunk <span>&lt;</span>frames_per_chunk<span>&gt;</span> \
  --overlap <span>&lt;</span>overlap_between_chunks<span>&gt;</span> \
  --tile_num <span>&lt;</span>number_of_tiles<span>&gt;</span></pre></div>
<div dir="auto"><div class="markdown-heading" dir="auto"><h3 tabindex="-1" dir="auto" class="heading-element">Parameters</h3><a id="user-content-parameters" class="anchor" aria-label="Permalink: Parameters" href="#parameters"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-parameters" aria-label="Permalink: Parameters" href="#parameters"></a></div>
<ul dir="auto">
<li>
<p dir="auto"><code>pre_trained_path</code> (str): Path to the stable video diffusion pipeline weights.<br>
<strong>Default</strong>: <code>./weights/stable-video-diffusion-img2vid-xt-1-1</code></p>
</li>
<li>
<p dir="auto"><code>unet_path</code> (str): Path to the pre-trained UNet model weights.<br>
<strong>Default</strong>: <code>./weights/StereoCrafter</code></p>
</li>
<li>
<p dir="auto"><code>input_folder</code> (str): Folder containing input video files for processing.<br>
<strong>Default</strong>: <code>./output_splatted</code></p>
</li>
<li>
<p dir="auto"><code>output_folder</code> (str): Folder where processed videos will be saved.<br>
<strong>Default</strong>: <code>./completed_output</code></p>
</li>
<li>
<p dir="auto"><code>frames_chunk</code> (int): Number of frames to process per chunk.<br>
<strong>Default</strong>: <code>23</code></p>
</li>
<li>
<p dir="auto"><code>overlap</code> (int): Overlap between consecutive chunks.<br>
<strong>Default</strong>: <code>3</code></p>
</li>
<li>
<p dir="auto"><code>tile_num</code> (int): Number of tiles for spatial processing.<br>
<strong>Default</strong>: <code>2</code></p>
</li>
</ul>
<div dir="auto"><div class="markdown-heading" dir="auto"><h3 tabindex="-1" dir="auto" class="heading-element">Examples</h3><a id="user-content-examples" class="anchor" aria-label="Permalink: Examples" href="#examples"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-examples" aria-label="Permalink: Examples" href="#examples"></a></div>
<ol dir="auto">
<li>
<p dir="auto"><strong>Process Videos Using Default Settings</strong></p>
<div dir="auto"><pre>python inpainting.py</pre></div>
</li>
<li>
<p dir="auto"><strong>Process Using Custom Settings</strong></p>
<div dir="auto"><pre>python inpainting.py \
  --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
  --unet_path ./weights/StereoCrafter \
  --input_folder ./input_videos_path \
  --output_folder ./output_videos_path \
  --frames_chunk 23 \
  --overlap 3 \
  --tile_num 2</pre></div>
</li>
</ol>
<div dir="auto"><div class="markdown-heading" dir="auto"><h2 tabindex="-1" dir="auto" class="heading-element">Notes</h2><a id="user-content-notes" class="anchor" aria-label="Permalink: Notes" href="#notes"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-notes" aria-label="Permalink: Notes" href="#notes"></a></div>
<ul dir="auto">
<li><strong>Input Videos</strong>: Ensure input videos are in <code>.mp4</code> format and have reasonable resolution for processing.</li>
<li><strong>GPU Usage</strong>: Inpainting tasks are GPU-intensive. Ensure sufficient GPU memory is available for processing.</li>
<li><strong>Output Structure</strong>: Processed videos will be saved in the specified <code>output_folder</code> with <code>_inpainting_results</code> suffix.</li>
</ul>
<div dir="auto"><div class="markdown-heading" dir="auto"><h2 tabindex="-1" dir="auto" class="heading-element">Troubleshooting</h2><a id="user-content-troubleshooting" class="anchor" aria-label="Permalink: Troubleshooting" href="#troubleshooting"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div><a id="user-content-troubleshooting" aria-label="Permalink: Troubleshooting" href="#troubleshooting"></a></div>
<ul dir="auto">
<li><strong>Missing Models</strong>: Ensure that the UNet and pipeline weights are placed in their respective directories.</li>
<li><strong>Dependencies Issues</strong>: Verify all required Python libraries are installed. Use <code>pip install -r requirements.txt</code> to install missing packages.</li>
<li><strong>Performance</strong>: Adjust <code>frames_chunk</code> and <code>tile_num</code> based on hardware capabilities to optimize performance and memory usage.</li>
</ul>
</div></section></div>
