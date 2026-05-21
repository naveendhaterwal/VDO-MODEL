import os
import logging
from typing import List
try:
    from moviepy import VideoFileClip, concatenate_videoclips
except ImportError:
    from moviepy.editor import VideoFileClip, concatenate_videoclips

logger = logging.getLogger(__name__)


class FFmpegAssemblyPipeline:
    def __init__(self, target_fps: int = 16, codec: str = "libx264"):
        self.target_fps = target_fps
        self.codec = codec

    def assemble(self, video_paths: List[str], output_path: str):
        logger.info(f"Assembling {len(video_paths)} clips into {output_path}")
        clips = []

        try:
            for path in video_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing required clip: {path}")

                clip = VideoFileClip(path)
                if clip.fps != self.target_fps:
                    if hasattr(clip, "with_fps"):
                        clip = clip.with_fps(self.target_fps)
                    else:
                        clip = clip.set_fps(self.target_fps)
                clips.append(clip)

            if not clips:
                raise ValueError("No valid clips found for assembly.")

            final_clip = concatenate_videoclips(clips, method="chain")

            logger.info("Writing final video file...")
            final_clip.write_videofile(
                output_path,
                fps=self.target_fps,
                codec=self.codec,
                audio=False,
                logger=None,
            )

        except Exception as e:
            logger.error(f"FFmpeg assembly failed: {e}")
            raise

        finally:
            for clip in clips:
                try:
                    clip.close()
                except Exception:
                    pass
            try:
                if "final_clip" in locals():
                    final_clip.close()
            except Exception:
                pass

        logger.info("FFmpeg assembly completed successfully.")
