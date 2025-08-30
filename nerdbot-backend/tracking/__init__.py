"""
Tracking module for pan/tilt control
"""
from .optimized_tracker import OptimizedTracker, PredictiveTracker

__all__ = ['OptimizedTracker', 'PredictiveTracker']