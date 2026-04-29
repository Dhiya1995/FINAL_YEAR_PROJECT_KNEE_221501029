"""
Reports package for Knee Osteoarthritis Classification System

This package contains:
- radiology_report: AI-powered radiology report generation using Med-Gemma LLM
- prescriptive_report: Personalized care plan generation using Gemini API
- pdf_generator: PDF report generation and formatting
"""

from .radiology_report import (
    RadiologyReportGenerator,
    generate_radiology_report
)

from .prescriptive_report import (
    PrescriptiveReportGenerator,
    generate_prescriptive_report
)

from .pdf_generator import (
    create_pdf_report,
    create_prescriptive_pdf
)

__all__ = [
    'RadiologyReportGenerator',
    'generate_radiology_report',
    'PrescriptiveReportGenerator',
    'generate_prescriptive_report',
    'create_pdf_report',
    'create_prescriptive_pdf'
]


