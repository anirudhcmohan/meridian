import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BriefsMonitoring:
    def __init__(self):
        self.metrics = {
            'briefs_generated': 0,
            'briefs_failed': 0,
            'articles_processed': 0,
            'clustering_operations': 0,
            'llm_calls': 0,
            'llm_failures': 0,
            'last_successful_brief': None,
            'uptime_start': datetime.now(),
            'total_processing_time': 0.0
        }
        
    def increment_briefs_generated(self):
        self.metrics['briefs_generated'] += 1
        self.metrics['last_successful_brief'] = datetime.now()
        
    def increment_briefs_failed(self):
        self.metrics['briefs_failed'] += 1
        
    def increment_articles_processed(self, count: int = 1):
        self.metrics['articles_processed'] += count
        
    def increment_clustering_operations(self):
        self.metrics['clustering_operations'] += 1
        
    def increment_llm_calls(self):
        self.metrics['llm_calls'] += 1
        
    def increment_llm_failures(self):
        self.metrics['llm_failures'] += 1
        
    def add_processing_time(self, duration: float):
        self.metrics['total_processing_time'] += duration
        
    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self.metrics,
            'uptime_hours': (datetime.now() - self.metrics['uptime_start']).total_seconds() / 3600,
            'last_brief_hours_ago': (
                (datetime.now() - self.metrics['last_successful_brief']).total_seconds() / 3600
                if self.metrics['last_successful_brief'] else None
            ),
            'success_rate': (
                self.metrics['briefs_generated'] / 
                max(1, self.metrics['briefs_generated'] + self.metrics['briefs_failed'])
            ) * 100,
            'llm_success_rate': (
                (self.metrics['llm_calls'] - self.metrics['llm_failures']) / 
                max(1, self.metrics['llm_calls'])
            ) * 100
        }
        
    def get_health_status(self) -> Dict[str, Any]:
        metrics = self.get_metrics()
        
        # Determine health status
        is_healthy = True
        issues = []
        
        # Check if we haven't generated a brief in too long
        if metrics['last_brief_hours_ago'] and metrics['last_brief_hours_ago'] > 24:
            is_healthy = False
            issues.append(f"No brief generated in {metrics['last_brief_hours_ago']:.1f} hours")
            
        # Check success rates
        if metrics['success_rate'] < 80:
            is_healthy = False
            issues.append(f"Low brief success rate: {metrics['success_rate']:.1f}%")
            
        if metrics['llm_success_rate'] < 90:
            is_healthy = False
            issues.append(f"Low LLM success rate: {metrics['llm_success_rate']:.1f}%")
            
        return {
            'status': 'healthy' if is_healthy else 'unhealthy',
            'issues': issues,
            **metrics
        }

# Global monitoring instance
monitoring = BriefsMonitoring()

def log_error(context: str, error: Exception, details: Optional[Dict] = None):
    """Log an error with context and optional details."""
    logger.error(f"ERROR in {context}: {str(error)}")
    if details:
        logger.error(f"Details: {details}")
        
def log_info(context: str, message: str, details: Optional[Dict] = None):
    """Log an info message with context and optional details."""
    logger.info(f"{context}: {message}")
    if details:
        logger.info(f"Details: {details}")
        
def log_warning(context: str, message: str, details: Optional[Dict] = None):
    """Log a warning message with context and optional details."""
    logger.warning(f"{context}: {message}")
    if details:
        logger.warning(f"Details: {details}")

class Timer:
    """Context manager for timing operations."""
    def __init__(self, description: str):
        self.description = description
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        monitoring.add_processing_time(duration)
        log_info('timer', f"{self.description} completed in {duration:.2f}s")