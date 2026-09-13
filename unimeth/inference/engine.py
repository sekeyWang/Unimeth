"""Inference input setup and multiprocessing pipeline entry point."""

import logging


logger = logging.getLogger(__name__)


class InferenceEngine:
    """Prepare BAM-primary inputs and run the sole inference pipeline."""

    def __init__(self, args):
        self.args = args

    def _prepare_signal_routing(self) -> None:
        """Resolve BAM semantics and prepare signal-file routing."""
        from unimeth.ioutils.reader.bam_stream import resolve_bam_mode_from_path
        from unimeth.ioutils.reader.raw_signal import collect_signal_paths
        from unimeth.ioutils.reader.signal_index import (
            prepare_signal_routing,
            resolve_signal_index_path,
        )

        signal_paths = collect_signal_paths(
            self.args.signal_dir,
            suffixes=getattr(self.args, "signal_suffixes", None),
            label=getattr(self.args, "signal_label", None),
        )
        index_path = None
        if len(signal_paths) > 1:
            index_path = resolve_signal_index_path(
                self.args.signal_dir,
                getattr(self.args, "signal_index", None),
            )

        requested_mode = getattr(self.args, "bam_mode", "auto")
        resolved_mode, auto_detected = resolve_bam_mode_from_path(
            self.args.bam_dir,
            requested_mode=requested_mode,
            threads=1,
        )
        self.args.resolved_bam_mode = resolved_mode
        source = "auto-detected" if auto_detected else "configured"
        logger.info("BAM mode: %s (%s)", resolved_mode, source)

        def log_index_progress(progress) -> None:
            if progress.files_completed == 0:
                logger.info(
                    "Building signal route index for %s signal files...",
                    f"{progress.file_count:,}",
                )

        plan = prepare_signal_routing(
            signal_paths,
            index_path=index_path,
            progress_callback=log_index_progress,
        )
        self.args.signal_routing_plan = plan

        if plan.uses_index:
            stats = plan.index_stats
            action = "reused" if stats.reused else "built"
            logger.info(
                "Signal route index %s: %s "
                "(%s read(s) across %s file(s) in %.2fs)",
                action,
                stats.index_path,
                f"{stats.read_count:,}",
                f"{stats.file_count:,}",
                stats.elapsed_seconds,
            )
        else:
            logger.info("Signal routing: direct lookup in one signal file")

    def run(self, output_format: str = "bam"):
        """Run BAM-primary streaming inference."""
        from unimeth.inference.multiprocess_pipeline import (
            MultiprocessInferencePipeline,
        )

        pipeline = MultiprocessInferencePipeline(self.args, output_format)
        pipeline.validate_launch()
        self._prepare_signal_routing()
        return pipeline.run()
