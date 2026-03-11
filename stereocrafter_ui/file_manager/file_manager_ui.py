"""
File Manager WebUI Component - Phase 1 (MVP)
Centralized file management for all StereoCrafter pipeline folders
"""

import os
import glob
import shutil
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import gradio as gr
from pathlib import Path

class FileManagerUI:
    """File Manager for StereoCrafter pipeline folders"""
    
    # Define all pipeline folders
    FOLDERS = {
        "Source Videos": "./source_video/",
        "Input Videos": "./input_source_clips/",
        "Depth Maps": "./output_depthmaps/",
        "Splatted (Low-Res)": "./output_splatted/lowres/",
        "Splatted (Hi-Res)": "./output_splatted/hires/",
        "Inpainted Output": "./completed_output/",
        "Final Videos": "./final_videos/",
        "---": None,  # Separator
        "Source Videos (Finished)": "./source_video/finished/",
        "Input Videos (Finished)": "./input_source_clips/finished/",
        "Depth Maps (Finished)": "./output_depthmaps/finished/",
        "Splatted Low-Res (Finished)": "./output_splatted/lowres/finished/",
        "Splatted Hi-Res (Finished)": "./output_splatted/hires/finished/",
        "Inpainted (Finished)": "./completed_output/finished/",
        "Final Videos (Finished)": "./final_videos/finished/",
    }
    
    def __init__(self):
        self.current_folder = "./input_source_clips/"
        self.selected_files = []
    
    def format_size(self, size_bytes: int) -> str:
        """Convert bytes to human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def format_date(self, timestamp: float) -> str:
        """Convert timestamp to readable date"""
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
    
    def scan_folder(self, folder_name: str) -> Tuple[List[List], str]:
        """
        Scan folder and return file list with metadata
        Returns: (file_data, stats_text)
        """
        if folder_name == "---":
            return [], "Please select a folder"
        
        folder_path = self.FOLDERS.get(folder_name, "./")
        
        if not os.path.exists(folder_path):
            return [], f"❌ Folder does not exist: {folder_path}"
        
        # Get all files (videos and related files)
        all_files = []
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        other_extensions = ('.json', '.sidecar', '.spsidecar', '.fssidecar', '.txt')
        
        for ext in video_extensions + other_extensions:
            all_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        
        # Sort by modification time (newest first)
        all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Build file data list
        file_data = []
        total_size = 0
        
        for file_path in all_files:
            try:
                filename = os.path.basename(file_path)
                size = os.path.getsize(file_path)
                modified = os.path.getmtime(file_path)
                
                # Determine status
                status = "Active"
                if "finished" in folder_path.lower():
                    status = "Finished"
                
                # Add icon based on file type
                if filename.endswith(video_extensions):
                    icon = "🎬"
                elif filename.endswith(('.json', '.sidecar', '.spsidecar', '.fssidecar')):
                    icon = "📄"
                else:
                    icon = "📁"
                
                file_data.append([
                    False,  # Checkbox (not selected by default)
                    f"{icon} {filename}",
                    self.format_size(size),
                    self.format_date(modified),
                    status
                ])
                
                total_size += size
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Generate statistics
        stats_text = self._generate_stats(folder_path, len(file_data), total_size)
        
        return file_data, stats_text
    
    def _generate_stats(self, folder_path: str, file_count: int, total_size: int) -> str:
        """Generate statistics text for the folder"""
        # Count finished files if in active folder
        finished_count = 0
        active_count = file_count
        
        if "finished" not in folder_path.lower():
            finished_folder = os.path.join(folder_path, "finished")
            if os.path.exists(finished_folder):
                finished_files = glob.glob(os.path.join(finished_folder, "*.*"))
                finished_count = len(finished_files)
        
        # Get disk space
        try:
            import shutil as sh
            disk_usage = sh.disk_usage(folder_path)
            free_space = self.format_size(disk_usage.free)
        except:
            free_space = "Unknown"
        
        stats = f"""📊 Folder Statistics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 Current Folder: {folder_path}
📦 Total Files: {file_count}
💾 Total Size: {self.format_size(total_size)}
✅ Active Files: {active_count}
📥 Finished Files: {finished_count}
💿 Free Disk Space: {free_space}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        return stats
    
    def select_all_files(self, file_data) -> List[List]:
        """Select all files in the list"""
        # Handle pandas DataFrame from Gradio
        if hasattr(file_data, 'empty'):  # It's a pandas DataFrame
            if file_data.empty:
                return file_data
            # Convert to list of lists and set all checkboxes to True
            result = file_data.values.tolist()
            for row in result:
                row[0] = True
            return result
        
        # Handle list of lists
        if not file_data or len(file_data) == 0:
            return file_data
        
        return [[True] + row[1:] for row in file_data]
    
    def deselect_all_files(self, file_data) -> List[List]:
        """Deselect all files in the list"""
        # Handle pandas DataFrame from Gradio
        if hasattr(file_data, 'empty'):  # It's a pandas DataFrame
            if file_data.empty:
                return file_data
            # Convert to list of lists and set all checkboxes to False
            result = file_data.values.tolist()
            for row in result:
                row[0] = False
            return result
        
        # Handle list of lists
        if not file_data or len(file_data) == 0:
            return file_data
        
        return [[False] + row[1:] for row in file_data]
    
    def select_finished_files(self, file_data) -> List[List]:
        """Select only finished files"""
        # Handle pandas DataFrame from Gradio
        if hasattr(file_data, 'empty'):  # It's a pandas DataFrame
            if file_data.empty:
                return file_data
            # Convert to list of lists
            result = file_data.values.tolist()
            for row in result:
                # Select if status (column 4) is "Finished"
                row[0] = (row[4] == "Finished")
            return result
        
        # Handle list of lists
        if not file_data or len(file_data) == 0:
            return file_data
        
        result = []
        for row in file_data:
            # Select if status is "Finished"
            selected = row[4] == "Finished"
            result.append([selected] + row[1:])
        
        return result
    
    def get_selected_files(self, folder_name: str, file_data) -> List[str]:
        """Get list of selected file paths"""
        if folder_name == "---":
            return []
        
        # Handle pandas DataFrame from Gradio
        if hasattr(file_data, 'empty'):  # It's a pandas DataFrame
            if file_data.empty:
                return []
            file_data = file_data.values.tolist()
        
        # Handle empty list
        if not file_data or len(file_data) == 0:
            return []
        
        folder_path = self.FOLDERS.get(folder_name, "./")
        selected = []
        
        for row in file_data:
            if row[0]:  # If checkbox is True
                # Extract filename (remove icon)
                filename = row[1].split(" ", 1)[1] if " " in row[1] else row[1]
                file_path = os.path.join(folder_path, filename)
                if os.path.exists(file_path):
                    selected.append(file_path)
        
        return selected
    
    def move_files(self, folder_name: str, file_data: List[List], destination: str) -> Tuple[List[List], str, str]:
        """
        Move selected files to destination
        Returns: (updated_file_data, stats_text, status_message)
        """
        selected_files = self.get_selected_files(folder_name, file_data)
        
        if not selected_files:
            return file_data, "", "⚠️ No files selected"
        
        folder_path = self.FOLDERS.get(folder_name, "./")
        
        # Determine destination path
        if destination == "finished/":
            dest_path = os.path.join(folder_path, "finished")
        elif destination == "parent/":
            dest_path = os.path.dirname(folder_path.rstrip('/'))
        elif destination.startswith("→ "):
            # Pipeline folder destination
            dest_folder_name = destination[2:]  # Remove "→ " prefix
            dest_path = self.FOLDERS.get(dest_folder_name)
            if not dest_path:
                return file_data, "", f"❌ Invalid destination: {dest_folder_name}"
        else:
            return file_data, "", "❌ Invalid destination"
        
        # Create destination folder if it doesn't exist
        os.makedirs(dest_path, exist_ok=True)
        
        # Move files
        moved_count = 0
        failed_files = []
        
        for file_path in selected_files:
            try:
                filename = os.path.basename(file_path)
                dest_file = os.path.join(dest_path, filename)
                
                # Check if destination file exists
                if os.path.exists(dest_file):
                    # Skip or overwrite? For now, skip
                    failed_files.append(f"{filename} (already exists)")
                    continue
                
                shutil.move(file_path, dest_file)
                moved_count += 1
            except Exception as e:
                failed_files.append(f"{filename} ({str(e)})")
        
        # Generate status message
        status_msg = f"✅ Moved {moved_count} file(s) to {dest_path}"
        if failed_files:
            status_msg += f"\n❌ Failed: {len(failed_files)} file(s)\n" + "\n".join(failed_files[:5])
            if len(failed_files) > 5:
                status_msg += f"\n... and {len(failed_files) - 5} more"
        
        # Refresh file list
        new_file_data, new_stats = self.scan_folder(folder_name)
        
        return new_file_data, new_stats, status_msg
    
    def delete_files(self, folder_name: str, file_data: List[List]) -> Tuple[List[List], str, str]:
        """
        Delete selected files
        Returns: (updated_file_data, stats_text, status_message)
        """
        selected_files = self.get_selected_files(folder_name, file_data)
        
        if not selected_files:
            return file_data, "", "⚠️ No files selected"
        
        # Delete files
        deleted_count = 0
        failed_files = []
        
        for file_path in selected_files:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                filename = os.path.basename(file_path)
                failed_files.append(f"{filename} ({str(e)})")
        
        # Generate status message
        status_msg = f"✅ Deleted {deleted_count} file(s)"
        if failed_files:
            status_msg += f"\n❌ Failed: {len(failed_files)} file(s)\n" + "\n".join(failed_files[:5])
            if len(failed_files) > 5:
                status_msg += f"\n... and {len(failed_files) - 5} more"
        
        # Refresh file list
        new_file_data, new_stats = self.scan_folder(folder_name)
        
        return new_file_data, new_stats, status_msg

    def download_file(self, folder_name: str, file_data: List[List]) -> Tuple[str, str]:
        """
        Download selected file(s). If multiple files selected, creates a zip.
        Returns: (file_path_or_none, status_message)
        """
        import zipfile
        import tempfile
        from datetime import datetime
        
        selected_files = self.get_selected_files(folder_name, file_data)

        if not selected_files:
            return None, "⚠️ No file selected for download"

        # Single file download
        if len(selected_files) == 1:
            file_path = selected_files[0]
            filename = os.path.basename(file_path)

            if not os.path.exists(file_path):
                return None, f"❌ File not found: {filename}"

            return file_path, f"✅ Ready to download: {filename}"
        
        # Multiple files - create zip
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"download_{timestamp}.zip"
            zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in selected_files:
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))
            
            return zip_path, f"✅ Ready to download: {len(selected_files)} files as {zip_filename}"
        except Exception as e:
            return None, f"❌ Failed to create zip: {str(e)}"

    
    def clean_finished_files(self, folder_name: str) -> Tuple[List[List], str, str]:
        """
        Move all files to finished subfolder
        Returns: (updated_file_data, stats_text, status_message)
        """
        if folder_name == "---":
            return [], "", "⚠️ Please select a folder"
        
        if "finished" in folder_name.lower():
            return [], "", "⚠️ Already in a finished folder"
        
        folder_path = self.FOLDERS.get(folder_name, "./")
        finished_path = os.path.join(folder_path, "finished")
        
        if not os.path.exists(folder_path):
            return [], "", f"❌ Folder does not exist: {folder_path}"
        
        # Create finished folder
        os.makedirs(finished_path, exist_ok=True)
        
        # Get all files in the folder (not in subfolders)
        all_files = []
        for ext in ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.json', '.sidecar', '.spsidecar', '.fssidecar'):
            all_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        
        # Move files
        moved_count = 0
        failed_files = []
        
        for file_path in all_files:
            try:
                filename = os.path.basename(file_path)
                dest_file = os.path.join(finished_path, filename)
                
                if os.path.exists(dest_file):
                    failed_files.append(f"{filename} (already exists)")
                    continue
                
                shutil.move(file_path, dest_file)
                moved_count += 1
            except Exception as e:
                failed_files.append(f"{os.path.basename(file_path)} ({str(e)})")
        
        # Generate status message
        status_msg = f"✅ Moved {moved_count} file(s) to finished folder"
        if failed_files:
            status_msg += f"\n❌ Failed: {len(failed_files)} file(s)"
        
        # Refresh file list
        new_file_data, new_stats = self.scan_folder(folder_name)
        
        return new_file_data, new_stats, status_msg
    
    def restore_finished_files(self, folder_name: str) -> Tuple[List[List], str, str]:
        """
        Move all files from finished subfolder back to parent
        Returns: (updated_file_data, stats_text, status_message)
        """
        if folder_name == "---":
            return [], "", "⚠️ Please select a folder"
        
        folder_path = self.FOLDERS.get(folder_name, "./")
        
        # Determine source and destination
        if "finished" in folder_name.lower():
            # Already in finished folder, move to parent
            source_path = folder_path
            dest_path = os.path.dirname(folder_path.rstrip('/'))
        else:
            # In active folder, move from finished subfolder
            source_path = os.path.join(folder_path, "finished")
            dest_path = folder_path
        
        if not os.path.exists(source_path):
            return [], "", f"❌ Source folder does not exist: {source_path}"
        
        # Get all files
        all_files = []
        for ext in ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.json', '.sidecar', '.spsidecar', '.fssidecar'):
            all_files.extend(glob.glob(os.path.join(source_path, f"*{ext}")))
        
        if not all_files:
            return [], "", "⚠️ No files to restore"
        
        # Move files
        moved_count = 0
        failed_files = []
        
        for file_path in all_files:
            try:
                filename = os.path.basename(file_path)
                dest_file = os.path.join(dest_path, filename)
                
                if os.path.exists(dest_file):
                    failed_files.append(f"{filename} (already exists)")
                    continue
                
                shutil.move(file_path, dest_file)
                moved_count += 1
            except Exception as e:
                failed_files.append(f"{os.path.basename(file_path)} ({str(e)})")
        
        # Generate status message
        status_msg = f"✅ Restored {moved_count} file(s) from finished folder"
        if failed_files:
            status_msg += f"\n❌ Failed: {len(failed_files)} file(s)"
        
        # Refresh file list
        new_file_data, new_stats = self.scan_folder(folder_name)
        
        return new_file_data, new_stats, status_msg
    
    def create_interface(self):
        """Create the Gradio interface for File Manager"""
        
        with gr.Row():
            with gr.Column(scale=3):
                folder_dropdown = gr.Dropdown(
                    choices=[k for k in self.FOLDERS.keys()],
                    value="Input Videos",
                    label="📁 Select Folder",
                    info="Choose which pipeline folder to manage",
                    interactive=True
                )
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
        
        # Statistics display
        # Get initial stats
        _, initial_stats = self.scan_folder("Input Videos")
        
        stats_display = gr.Textbox(
            value=initial_stats,
            label="",
            lines=8,
            interactive=False,
            show_label=False
        )
        
        # File list table
        # Get initial data
        initial_data, _ = self.scan_folder("Input Videos")
        
        file_list = gr.Dataframe(
            value=initial_data,
            headers=["Select", "Filename", "Size", "Modified", "Status"],
            datatype=["bool", "str", "str", "str", "str"],
            col_count=(5, "fixed"),
            interactive=True,
            label="📋 Files",
            wrap=True
        )
        
        # Selection buttons
        with gr.Row():
            select_all_btn = gr.Button("☑️ Select All", size="sm")
            deselect_all_btn = gr.Button("☐ Deselect All", size="sm")
            select_finished_btn = gr.Button("📥 Select Finished", size="sm")
            download_btn = gr.Button("⬇️ Download", size="sm")
        
        gr.Markdown("💡 **Tip:** Check the boxes next to files you want to move or delete")
        
        # Download output (visible so user can click to download)
        download_file_output = gr.File(label="📥 Click filename below to download", visible=True)
        
        # Actions
        gr.Markdown("### 🔧 Actions")
        gr.Markdown("**Move Selected:** Moves checked files to the chosen destination")
        with gr.Row():
            with gr.Column(scale=2):
                move_to_dropdown = gr.Dropdown(
                    choices=[
                        "finished/",
                        "parent/",
                        "---",
                        "→ Source Videos",
                        "→ Input Videos",
                        "→ Depth Maps",
                        "→ Splatted (Low-Res)",
                        "→ Splatted (Hi-Res)",
                        "→ Inpainted Output",
                    ],
                    value="finished/",
                    label="Move To",
                    info="Quick move to pipeline folders or finished/parent",
                    interactive=True
                )
            with gr.Column(scale=1):
                move_btn = gr.Button("📦 Move Selected", variant="primary")
            with gr.Column(scale=1):
                delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")
        
        # Batch operations
        gr.Markdown("### 🧹 Batch Operations")
        gr.Markdown("""
**Clean Finished:** Moves ALL files in current folder to `finished/` subfolder (does NOT delete)  
**Restore All:** Moves ALL files from `finished/` back to parent folder (for reprocessing)
        """)
        with gr.Row():
            clean_btn = gr.Button("🧹 Clean Finished", size="sm")
            restore_btn = gr.Button("♻️ Restore All", size="sm")
        
        # Status
        status_label = gr.Textbox(
            label="Status",
            value="Ready",
            interactive=False,
            lines=3
        )
        
        # Event handlers
        
        # Initial load function
        def initial_load():
            file_data, stats = self.scan_folder("Input Videos")
            return file_data, stats, "Ready"
        
        # Refresh folder
        def refresh_folder(folder_name):
            file_data, stats = self.scan_folder(folder_name)
            return file_data, stats, "✅ Refreshed"
        
        folder_dropdown.change(
            fn=refresh_folder,
            inputs=[folder_dropdown],
            outputs=[file_list, stats_display, status_label]
        )
        
        refresh_btn.click(
            fn=refresh_folder,
            inputs=[folder_dropdown],
            outputs=[file_list, stats_display, status_label]
        )
        
        # Selection buttons
        select_all_btn.click(
            fn=self.select_all_files,
            inputs=[file_list],
            outputs=[file_list]
        )
        
        deselect_all_btn.click(
            fn=self.deselect_all_files,
            inputs=[file_list],
            outputs=[file_list]
        )
        
        select_finished_btn.click(
            fn=self.select_finished_files,
            inputs=[file_list],
            outputs=[file_list]
        )
        
        # Move files
        move_btn.click(
            fn=self.move_files,
            inputs=[folder_dropdown, file_list, move_to_dropdown],
            outputs=[file_list, stats_display, status_label]
        )
        
        # Delete files
        delete_btn.click(
            fn=self.delete_files,
            inputs=[folder_dropdown, file_list],
            outputs=[file_list, stats_display, status_label]
        )
        
        # Download file
        download_btn.click(
            fn=self.download_file,
            inputs=[folder_dropdown, file_list],
            outputs=[download_file_output, status_label]
        )
        
        # Batch operations
        clean_btn.click(
            fn=self.clean_finished_files,
            inputs=[folder_dropdown],
            outputs=[file_list, stats_display, status_label]
        )
        
        restore_btn.click(
            fn=self.restore_finished_files,
            inputs=[folder_dropdown],
            outputs=[file_list, stats_display, status_label]
        )
        
        return folder_dropdown, file_list, stats_display, status_label
