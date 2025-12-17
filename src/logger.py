"""
Logging Framework for PMB UNSIQ Dataset Generation Pipeline
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
from colorama import Fore, Style, init

# Initialize colorama for Windows support
init()


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


class PipelineLogger:
    """
    Logging utility for the dataset generation pipeline.
    Provides structured logging for phases, batches, and metrics.
    """
    
    def __init__(self, name: str = "pipeline", log_file: Optional[str] = None, 
                 level: str = "INFO"):
        """
        Initialize the pipeline logger.
        
        Args:
            name: Logger name
            log_file: Path to log file (optional)
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(console_handler)
        
        # File handler (no colors)
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:  # Only makedirs if there's a directory component
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def log_phase_start(self, phase: int, name: str):
        """Log the start of a pipeline phase"""
        separator = "=" * 60
        self.logger.info(separator)
        self.logger.info(f"PHASE {phase}: {name.upper()} - STARTED")
        self.logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(separator)
    
    def log_phase_end(self, phase: int, name: str, stats: Dict[str, Any] = None):
        """Log the end of a pipeline phase with statistics"""
        separator = "=" * 60
        self.logger.info(separator)
        self.logger.info(f"PHASE {phase}: {name.upper()} - COMPLETED")
        if stats:
            self.logger.info("Statistics:")
            for key, value in stats.items():
                self.logger.info(f"  • {key}: {value}")
        self.logger.info(separator)
    
    def log_batch(self, batch_id: int, total_batches: int, 
                  generated: int, total: int, speed: float = None, 
                  memory_gb: float = None):
        """Log batch progress during generation"""
        msg = f"[Batch {batch_id}/{total_batches}] Generated: {generated} | Total: {total}"
        if speed:
            msg += f" | Speed: {speed:.0f} tok/s"
        if memory_gb:
            msg += f" | GPU: {memory_gb:.1f}GB"
        self.logger.info(msg)
    
    def log_validation_progress(self, current: int, total: int, 
                                 accepted: int, rejected: int):
        """Log validation progress"""
        accept_rate = (accepted / current * 100) if current > 0 else 0
        self.logger.info(
            f"Validated: {current}/{total} | "
            f"Accepted: {accepted} ({accept_rate:.1f}%) | "
            f"Rejected: {rejected}"
        )
    
    def log_error_with_context(self, phase: int, error: str, 
                                context: Dict[str, Any] = None):
        """Log error with additional context"""
        self.logger.error(f"Phase {phase} Error: {error}")
        if context:
            self.logger.error(f"Context: {context}")
    
    def log_checkpoint(self, checkpoint_path: str, pairs_count: int):
        """Log checkpoint save"""
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path} ({pairs_count} pairs)")
    
    def log_metric(self, name: str, value: Any):
        """Log a single metric"""
        self.logger.info(f"📊 {name}: {value}")


class MetricsTracker:
    """
    Track and aggregate metrics across the pipeline.
    """
    
    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def start_pipeline(self):
        """Record pipeline start time"""
        self.start_time = datetime.now()
    
    def end_pipeline(self):
        """Record pipeline end time"""
        self.end_time = datetime.now()
    
    def record_phase(self, phase: int, phase_name: str,
                     start_time: datetime, end_time: datetime,
                     stats: Dict[str, Any]):
        """Record metrics for a phase"""
        self.metrics[f"phase_{phase}"] = {
            "name": phase_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "stats": stats
        }
    
    def add_metric(self, phase: int, key: str, value: Any):
        """Add a metric to a phase"""
        phase_key = f"phase_{phase}"
        if phase_key not in self.metrics:
            self.metrics[phase_key] = {"stats": {}}
        self.metrics[phase_key]["stats"][key] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall summary of pipeline metrics"""
        total_duration = 0
        if self.start_time and self.end_time:
            total_duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            "pipeline_start": self.start_time.isoformat() if self.start_time else None,
            "pipeline_end": self.end_time.isoformat() if self.end_time else None,
            "total_duration_seconds": total_duration,
            "total_duration_human": self._format_duration(total_duration),
            "phases": self.metrics
        }
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file"""
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)


# Create default logger instance
def get_logger(name: str = "pipeline", log_file: str = None) -> PipelineLogger:
    """Get a pipeline logger instance"""
    return PipelineLogger(name=name, log_file=log_file)
