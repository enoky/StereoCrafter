"""
StereoCrafter Combined WebUI
A unified interface for depth estimation, splatting, inpainting, and merging operations.

This is the main entry point that orchestrates all UI components.
"""

import logging
import gradio as gr
from core.common.gpu_utils import check_cuda_availability, CUDA_AVAILABLE

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Import UI components from the modular structure
from stereocrafter_ui.depthcrafter import DepthCrafterWebUI
from stereocrafter_ui.splatting import SplatterWebUI
from stereocrafter_ui.inpainting import InpaintingWebUI
from stereocrafter_ui.merging import MergingWebUI
from stereocrafter_ui.file_manager import FileManagerUI


class CombinedWebUI:
    """
    Main orchestrator for the StereoCrafter WebUI.
    Combines all UI components into a single tabbed interface.
    """
    
    def __init__(self):
        # Initialize all components
        self.depthcrafter_gui = DepthCrafterWebUI()
        self.splatting_gui = SplatterWebUI()
        self.inpainting_gui = InpaintingWebUI()
        self.merging_gui = MergingWebUI()
        self.file_manager_gui = FileManagerUI()
        
    def create_interface(self):
        """Creates the combined Gradio interface with all tabs"""
        with gr.Blocks(title="StereoCrafter Combined WebUI") as interface:
            gr.Markdown("# StereoCrafter Combined WebUI")
            gr.Markdown("A unified interface for depth estimation, splatting, inpainting, and merging operations.")
            
            with gr.Tab("DepthCrafter"):
                self.depthcrafter_gui.create_interface()
                
            with gr.Tab("Splatting"):
                self.splatting_gui.create_interface()
                
            with gr.Tab("Inpainting"):
                self.inpainting_gui.create_interface()
                
            with gr.Tab("Merging"):
                self.merging_gui.create_interface()
            
            with gr.Tab("📂 File Manager"):
                self.file_manager_gui.create_interface()
        
        return interface


def launch():
    """Launch the combined WebUI"""
    app = CombinedWebUI()
    interface = app.create_interface()
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    # Check CUDA availability
    cuda_available = check_cuda_availability()
    print(f"[DEBUG] CUDA_AVAILABLE set to: {cuda_available}")
    launch()
