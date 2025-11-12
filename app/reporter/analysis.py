"""
Print analysis module for problem detection and visualization.

This module analyzes simulation results from the hybrid material state
to identify:
- Unsupported segments (bridging)
- Excessive droop
- Poor layer fusion
- Temperature issues

It can also generate color-coded meshes for visualization.
"""

import logging
import numpy as np

from app.physics.mechanics.droop_detection import DroopDetector


logger = logging.getLogger(__name__)


class PrintAnalysis:
    """Analyzes simulation results for problems and visualizes them"""
    
    def __init__(self, material_type='PLA'):
        """
        Initialize print analysis.
        
        Parameters:
        -----------
        material_type : str
            Material type for context-specific analysis
        """
        self.material_type = material_type
        self.droop_detector = DroopDetector(material_type)
    
    def analyze_simulation(self, physics_data):
        """
        Analyze complete simulation results.
        
        Parameters:
        -----------
        physics_data : dict
            Physics simulation data from VoxelSpace.get_physics_simulation_data()
            
        Returns:
        --------
        dict
            Complete analysis report
        """
        if physics_data is None:
            logger.warning("No physics data available - physics simulation not enabled")
            return {'error': 'Physics simulation not enabled'}
        
        segments = physics_data['segments']
        stats = physics_data['statistics']
        
        # Run batch analysis on all segments
        droop_summary = self.droop_detector.analyze_segments_batch(segments)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(droop_summary, stats)
        
        # Create severity classification
        severity = self._classify_severity(droop_summary)
        
        report = {
            'material_type': physics_data['material_type'],
            'thermal_enabled': physics_data['thermal_enabled'],
            'droop_enabled': physics_data['droop_enabled'],
            'statistics': stats,
            'droop_analysis': droop_summary,
            'recommendations': recommendations,
            'severity': severity,
            'printability_score': self._calculate_printability_score(droop_summary, stats),
        }
        
        return report
    
    def _generate_recommendations(self, droop_summary, stats):
        """
        Generate recommendations based on analysis.
        
        Returns:
        --------
        list
            List of recommendation strings
        """
        recommendations = []
        
        # Check for excessive droop
        if droop_summary['droop_stats']['severe_count'] > 0:
            recommendations.append(
                f"⚠️ {droop_summary['droop_stats']['severe_count']} segments with severe droop detected. "
                "Consider adding support structures or reducing bridge lengths."
            )
        
        if droop_summary['droop_stats']['max_droop'] > 1.0:
            recommendations.append(
                f"⚠️ Maximum droop of {droop_summary['droop_stats']['max_droop']:.2f}mm detected. "
                "Print may fail or have significant quality issues."
            )
        
        # Check bridge statistics
        if droop_summary['bridge_stats']['total_bridges'] > 0:
            longest = droop_summary['bridge_stats']['longest_bridge']
            if longest > 30:
                recommendations.append(
                    f"⚠️ Very long bridge detected ({longest:.1f}mm). "
                    "Consider splitting the model or adding support."
                )
            elif longest > 15:
                recommendations.append(
                    f"ℹ️ Long bridge detected ({longest:.1f}mm). "
                    "Ensure cooling fan is at 100% for bridges."
                )
        
        # Check unsupported percentage
        unsupported_pct = (stats['unsupported_segments'] / stats['total_segments'] * 100 
                          if stats['total_segments'] > 0 else 0)
        if unsupported_pct > 50:
            recommendations.append(
                f"⚠️ {unsupported_pct:.1f}% of segments are unsupported. "
                "Model has extensive bridging - consider redesign or heavy support."
            )
        elif unsupported_pct > 20:
            recommendations.append(
                f"ℹ️ {unsupported_pct:.1f}% of segments are unsupported. "
                "Some bridging required - tune bridge settings."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "✅ No major issues detected. Model appears printable."
            )
        
        return recommendations
    
    def _classify_severity(self, droop_summary):
        """
        Classify overall print severity.
        
        Returns:
        --------
        str
            'excellent', 'good', 'fair', 'poor', or 'critical'
        """
        severe_count = droop_summary['droop_stats']['severe_count']
        moderate_count = droop_summary['droop_stats']['moderate_count']
        max_droop = droop_summary['droop_stats']['max_droop']
        
        if severe_count > 0 or max_droop > 1.0:
            return 'critical'
        elif moderate_count > 10 or max_droop > 0.5:
            return 'poor'
        elif moderate_count > 5 or max_droop > 0.3:
            return 'fair'
        elif moderate_count > 0 or max_droop > 0.1:
            return 'good'
        else:
            return 'excellent'
    
    def _calculate_printability_score(self, droop_summary, stats):
        """
        Calculate a printability score from 0-100.
        
        Higher is better. Based on:
        - Droop severity
        - Support coverage
        - Bridge difficulty
        
        Returns:
        --------
        float
            Score from 0-100
        """
        score = 100.0
        
        # Penalize severe droop heavily
        severe_count = droop_summary['droop_stats']['severe_count']
        score -= severe_count * 10
        
        # Penalize moderate droop
        moderate_count = droop_summary['droop_stats']['moderate_count']
        score -= moderate_count * 2
        
        # Penalize based on maximum droop
        max_droop = droop_summary['droop_stats']['max_droop']
        if max_droop > 1.0:
            score -= 30
        elif max_droop > 0.5:
            score -= 15
        elif max_droop > 0.3:
            score -= 5
        
        # Penalize based on bridge difficulty
        if droop_summary['bridge_stats']['total_bridges'] > 0:
            longest = droop_summary['bridge_stats']['longest_bridge']
            if longest > 30:
                score -= 20
            elif longest > 15:
                score -= 10
        
        # Ensure score stays in range
        score = max(0.0, min(100.0, score))
        
        return score
    
    def generate_text_report(self, analysis_report):
        """
        Generate human-readable text report.
        
        Parameters:
        -----------
        analysis_report : dict
            Report from analyze_simulation()
            
        Returns:
        --------
        str
            Formatted text report
        """
        if 'error' in analysis_report:
            return f"Error: {analysis_report['error']}"
        
        lines = [
            "=" * 70,
            "VOLCO THERMAL & DROOP SIMULATION ANALYSIS",
            "=" * 70,
            "",
            f"Material Type: {analysis_report['material_type']}",
            f"Thermal Simulation: {'Enabled' if analysis_report['thermal_enabled'] else 'Disabled'}",
            f"Droop Simulation: {'Enabled' if analysis_report['droop_enabled'] else 'Disabled'}",
            "",
            "OVERALL ASSESSMENT:",
            f"  Severity Level: {analysis_report['severity'].upper()}",
            f"  Printability Score: {analysis_report['printability_score']:.1f}/100",
            "",
        ]
        
        # Add statistics
        stats = analysis_report['statistics']
        lines.extend([
            "STATISTICS:",
            f"  Total segments: {stats['total_segments']}",
            f"  Unsupported segments: {stats['unsupported_segments']}",
            f"  Solidified segments: {stats['solidified_segments']}",
            f"  Active (hot) segments: {stats['active_segments']}",
            f"  Simulation time: {stats['total_time']:.2f}s",
        ])
        
        # Add strand tracking stats if available
        if 'max_strand_length' in stats:
            lines.append(f"  Maximum continuous strand: {stats['max_strand_length']:.2f} mm")
        
        lines.append("")
        
        # Add droop analysis
        droop_stats = analysis_report['droop_analysis']['droop_stats']
        lines.extend([
            "DROOP ANALYSIS:",
            f"  Maximum droop: {droop_stats['max_droop']:.3f} mm",
            f"  Average droop: {droop_stats['avg_droop']:.3f} mm",
            f"  Severe droop issues: {droop_stats['severe_count']}",
            f"  Moderate droop issues: {droop_stats['moderate_count']}",
            "",
        ])
        
        # Add bridge statistics if any
        bridge_stats = analysis_report['droop_analysis']['bridge_stats']
        if bridge_stats['total_bridges'] > 0:
            lines.extend([
                "BRIDGE ANALYSIS:",
                f"  Total bridges: {bridge_stats['total_bridges']}",
                f"  Longest bridge: {bridge_stats['longest_bridge']:.2f} mm",
                f"  Average bridge length: {bridge_stats['avg_bridge_length']:.2f} mm",
                "",
            ])
        
        # Add recommendations
        lines.append("RECOMMENDATIONS:")
        for rec in analysis_report['recommendations']:
            lines.append(f"  {rec}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def get_segment_colors_for_visualization(self, segments):
        """
        Generate color values for segments based on their state.
        
        Color scheme:
        - Green: Good (no issues)
        - Yellow: Minor droop
        - Orange: Moderate droop or unsupported
        - Red: Severe droop
        - Purple: Critical droop
        
        Parameters:
        -----------
        segments : list[FilamentSegment]
            List of segments to color
            
        Returns:
        --------
        list
            List of RGB color tuples (0-1 range)
        """
        colors = []
        
        for segment in segments:
            analysis = self.droop_detector.analyze_segment(segment)
            severity = analysis['droop_severity']
            
            if severity == 'critical':
                color = (0.5, 0.0, 0.5)  # Purple
            elif severity == 'severe':
                color = (1.0, 0.0, 0.0)  # Red
            elif severity == 'moderate':
                color = (1.0, 0.5, 0.0)  # Orange
            elif severity == 'minor':
                color = (1.0, 1.0, 0.0)  # Yellow
            else:
                color = (0.0, 1.0, 0.0)  # Green
            
            colors.append(color)
        
        return colors
    
    def export_detailed_csv(self, segments, output_path):
        """
        Export detailed segment data to CSV for external analysis.
        
        Parameters:
        -----------
        segments : list[FilamentSegment]
            List of segments
        output_path : str
            Path to output CSV file
        """
        import csv
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'segment_id', 'start_x', 'start_y', 'start_z',
                'end_x', 'end_y', 'end_z', 'length', 'volume',
                'temperature', 'droop_offset', 'unsupported_length',
                'has_support', 'is_solidified', 'is_fused',
                'droop_severity', 'timestamp'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for seg in segments:
                analysis = self.droop_detector.analyze_segment(seg)
                writer.writerow({
                    'segment_id': seg.segment_id,
                    'start_x': seg.start_pos[0],
                    'start_y': seg.start_pos[1],
                    'start_z': seg.start_pos[2],
                    'end_x': seg.end_pos[0],
                    'end_y': seg.end_pos[1],
                    'end_z': seg.end_pos[2],
                    'length': seg.length,
                    'volume': seg.volume,
                    'temperature': seg.temperature,
                    'droop_offset': seg.droop_offset,
                    'unsupported_length': seg.unsupported_length,
                    'has_support': seg.support_below,
                    'is_solidified': seg.is_solidified,
                    'is_fused': seg.is_fused,
                    'droop_severity': analysis['droop_severity'],
                    'timestamp': seg.timestamp,
                })
        
        logger.info(f"Detailed segment data exported to {output_path}")
