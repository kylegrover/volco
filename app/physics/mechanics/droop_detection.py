"""
Droop detection and problem identification for unsupported filament.

This module analyzes filament segments to identify areas with excessive droop,
unsupported regions, and potential print quality issues.
"""

import math


class DroopDetector:
    """Detects and analyzes droop in printed structures"""
    
    # Thresholds for problem detection (mm)
    MINOR_DROOP_THRESHOLD = 0.1    # Barely noticeable
    MODERATE_DROOP_THRESHOLD = 0.3  # Visible but may be acceptable
    SEVERE_DROOP_THRESHOLD = 0.5    # Likely to cause problems
    CRITICAL_DROOP_THRESHOLD = 1.0  # Will definitely fail
    
    # Bridge length thresholds (mm)
    SHORT_BRIDGE = 5.0
    MEDIUM_BRIDGE = 15.0
    LONG_BRIDGE = 30.0
    
    def __init__(self, material_type='PLA'):
        """
        Initialize droop detector.
        
        Parameters:
        -----------
        material_type : str
            Material type for context-specific thresholds
        """
        self.material_type = material_type
    
    def classify_droop_severity(self, droop_amount):
        """
        Classify droop severity.
        
        Parameters:
        -----------
        droop_amount : float
            Droop amount in mm
            
        Returns:
        --------
        str
            Severity level: 'none', 'minor', 'moderate', 'severe', or 'critical'
        """
        if droop_amount < self.MINOR_DROOP_THRESHOLD:
            return 'none'
        elif droop_amount < self.MODERATE_DROOP_THRESHOLD:
            return 'minor'
        elif droop_amount < self.SEVERE_DROOP_THRESHOLD:
            return 'moderate'
        elif droop_amount < self.CRITICAL_DROOP_THRESHOLD:
            return 'severe'
        else:
            return 'critical'
    
    def classify_bridge_length(self, bridge_length):
        """
        Classify bridge difficulty.
        
        Parameters:
        -----------
        bridge_length : float
            Bridge length in mm
            
        Returns:
        --------
        str
            Bridge difficulty: 'short', 'medium', 'long', or 'extreme'
        """
        if bridge_length < self.SHORT_BRIDGE:
            return 'short'
        elif bridge_length < self.MEDIUM_BRIDGE:
            return 'medium'
        elif bridge_length < self.LONG_BRIDGE:
            return 'long'
        else:
            return 'extreme'
    
    def analyze_segment(self, segment):
        """
        Analyze a filament segment for droop and support issues.
        
        Parameters:
        -----------
        segment : FilamentSegment
            Filament segment to analyze
            
        Returns:
        --------
        dict
            Analysis results including problems and recommendations
        """
        analysis = {
            'has_support': segment.support_below is not None and segment.support_below,
            'droop_amount': segment.droop_offset,
            'droop_severity': self.classify_droop_severity(segment.droop_offset),
            'unsupported_length': segment.unsupported_length if hasattr(segment, 'unsupported_length') else 0,
            'is_solidified': segment.is_solidified,
            'temperature': segment.temperature,
            'problems': [],
            'warnings': [],
            'info': []
        }
        
        # Check for problems
        if not analysis['has_support']:
            bridge_class = self.classify_bridge_length(analysis['unsupported_length'])
            analysis['warnings'].append(
                f"Unsupported segment ({analysis['unsupported_length']:.1f}mm {bridge_class} bridge)"
            )
        
        if analysis['droop_severity'] in ['severe', 'critical']:
            analysis['problems'].append(
                f"Excessive droop detected: {analysis['droop_amount']:.2f}mm "
                f"({analysis['droop_severity']} level)"
            )
        elif analysis['droop_severity'] == 'moderate':
            analysis['warnings'].append(
                f"Moderate droop: {analysis['droop_amount']:.2f}mm - may affect appearance"
            )
        elif analysis['droop_severity'] == 'minor':
            analysis['info'].append(
                f"Minor droop: {analysis['droop_amount']:.2f}mm - likely acceptable"
            )
        
        # Check fusion state
        if not segment.is_solidified and hasattr(segment, 'is_fused') and not segment.is_fused:
            analysis['warnings'].append(
                "Material not yet fused - may indicate cooling issues"
            )
        
        return analysis
    
    def analyze_segments_batch(self, segments):
        """
        Analyze multiple segments and generate summary statistics.
        
        Parameters:
        -----------
        segments : list[FilamentSegment]
            List of segments to analyze
            
        Returns:
        --------
        dict
            Summary statistics and problem areas
        """
        total_segments = len(segments)
        
        summary = {
            'total_segments': total_segments,
            'unsupported_count': 0,
            'problem_segments': [],
            'warning_segments': [],
            'droop_stats': {
                'max_droop': 0.0,
                'avg_droop': 0.0,
                'severe_count': 0,
                'moderate_count': 0,
            },
            'bridge_stats': {
                'total_bridges': 0,
                'longest_bridge': 0.0,
                'avg_bridge_length': 0.0,
            }
        }
        
        total_droop = 0.0
        bridge_lengths = []
        
        for i, segment in enumerate(segments):
            analysis = self.analyze_segment(segment)
            
            # Update statistics
            if not analysis['has_support']:
                summary['unsupported_count'] += 1
                if hasattr(segment, 'unsupported_length'):
                    bridge_lengths.append(segment.unsupported_length)
            
            total_droop += analysis['droop_amount']
            summary['droop_stats']['max_droop'] = max(
                summary['droop_stats']['max_droop'],
                analysis['droop_amount']
            )
            
            if analysis['droop_severity'] in ['severe', 'critical']:
                summary['droop_stats']['severe_count'] += 1
                summary['problem_segments'].append((i, segment, analysis))
            elif analysis['droop_severity'] == 'moderate':
                summary['droop_stats']['moderate_count'] += 1
                summary['warning_segments'].append((i, segment, analysis))
            
            if analysis['problems']:
                if (i, segment, analysis) not in summary['problem_segments']:
                    summary['problem_segments'].append((i, segment, analysis))
            elif analysis['warnings']:
                if (i, segment, analysis) not in summary['warning_segments']:
                    summary['warning_segments'].append((i, segment, analysis))
        
        # Calculate averages
        if total_segments > 0:
            summary['droop_stats']['avg_droop'] = total_droop / total_segments
        
        if bridge_lengths:
            summary['bridge_stats']['total_bridges'] = len(bridge_lengths)
            summary['bridge_stats']['longest_bridge'] = max(bridge_lengths)
            summary['bridge_stats']['avg_bridge_length'] = sum(bridge_lengths) / len(bridge_lengths)
        
        return summary
    
    def generate_report(self, summary):
        """
        Generate a human-readable report from summary statistics.
        
        Parameters:
        -----------
        summary : dict
            Summary statistics from analyze_segments_batch
            
        Returns:
        --------
        str
            Formatted report text
        """
        report_lines = [
            "=" * 60,
            "DROOP AND SUPPORT ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Total segments analyzed: {summary['total_segments']}",
            f"Unsupported segments: {summary['unsupported_count']}",
            "",
            "DROOP STATISTICS:",
            f"  Maximum droop: {summary['droop_stats']['max_droop']:.2f} mm",
            f"  Average droop: {summary['droop_stats']['avg_droop']:.2f} mm",
            f"  Severe issues: {summary['droop_stats']['severe_count']}",
            f"  Moderate issues: {summary['droop_stats']['moderate_count']}",
            "",
        ]
        
        if summary['bridge_stats']['total_bridges'] > 0:
            report_lines.extend([
                "BRIDGE STATISTICS:",
                f"  Total bridges: {summary['bridge_stats']['total_bridges']}",
                f"  Longest bridge: {summary['bridge_stats']['longest_bridge']:.2f} mm",
                f"  Average bridge: {summary['bridge_stats']['avg_bridge_length']:.2f} mm",
                "",
            ])
        
        if summary['problem_segments']:
            report_lines.extend([
                "CRITICAL PROBLEMS:",
                "  (Segments requiring attention)",
            ])
            for i, segment, analysis in summary['problem_segments'][:10]:  # Show first 10
                report_lines.append(f"  Segment {i}: {', '.join(analysis['problems'])}")
            if len(summary['problem_segments']) > 10:
                report_lines.append(f"  ... and {len(summary['problem_segments']) - 10} more")
            report_lines.append("")
        
        if summary['warning_segments']:
            report_lines.extend([
                "WARNINGS:",
                "  (Segments that may have issues)",
            ])
            for i, segment, analysis in summary['warning_segments'][:5]:  # Show first 5
                report_lines.append(f"  Segment {i}: {', '.join(analysis['warnings'])}")
            if len(summary['warning_segments']) > 5:
                report_lines.append(f"  ... and {len(summary['warning_segments']) - 5} more")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
