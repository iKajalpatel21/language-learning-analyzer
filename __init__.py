"""
Language Learning Engagement Analyzer

A product analytics framework for understanding and improving
learner engagement in language learning applications.
"""

from .engagement_analyzer import (
    StreakAnalyzer,
    LearningFrictionAnalyzer,
    ABTestAnalyzer,
    generate_sample_data
)

__version__ = "1.0.0"
__author__ = "Kajal Patel"

__all__ = [
    "StreakAnalyzer",
    "LearningFrictionAnalyzer", 
    "ABTestAnalyzer",
    "generate_sample_data"
]
