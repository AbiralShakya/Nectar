#!/usr/bin/env python3
"""
Generate comprehensive summary report from synthetic test results.
Analyzes all test outputs and creates summary plots and metrics.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd

class SyntheticTestReportGenerator:
    """
    Generate comprehensive summary report from synthetic test results.
    """
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.report_data = {}
        
    def generate_report(self, output_file: str, generate_plots: bool = True):
        """Generate comprehensive summary report."""
        print("Generating Synthetic Test Summary Report...")
        
        # Load all test results
        self._load_all_results()
        
        # Analyze results
        self._analyze_results()
        
        # Generate summary
        summary = self._create_summary()
        
        # Save summary
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary report saved to {output_file}")
        
        # Generate plots if requested
        if generate_plots:
            self._generate_plots()
    
    def _load_all_results(self):
        """Load all test result files."""
        print("Loading test results...")
        
        # Load synthetic dataset config
        config_file = self.results_dir / "synthetic_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.report_data['synthetic_config'] = json.load(f)
        
        # Load dynamic adaptation results
        adaptation_file = self.results_dir / "dynamic_adaptation_results.json"
        if adaptation_file.exists():
            with open(adaptation_file, 'r') as f:
                self.report_data['dynamic_adaptation'] = json.load(f)
        
        # Load energy-aware TTT results
        self.report_data['energy_aware_ttt'] = {}
        for lambda_val in [0.001, 0.01, 0.05, 0.1, 0.2]:
            ttt_file = self.results_dir / f"energy_aware_ttt_lambda_{lambda_val}.json"
            if ttt_file.exists():
                with open(ttt_file, 'r') as f:
                    self.report_data['energy_aware_ttt'][str(lambda_val)] = json.load(f)
        
        # Load thermal-aware routing results
        thermal_file = self.results_dir / "thermal_aware_routing_results.json"
        if thermal_file.exists():
            with open(thermal_file, 'r') as f:
                self.report_data['thermal_aware_routing'] = json.load(f)
        
        # Load network topology results
        topology_file = self.results_dir / "network_topology_results.json"
        if topology_file.exists():
            with open(topology_file, 'r') as f:
                self.report_data['network_topology'] = json.load(f)
        
        # Load TTT validation results
        validation_file = self.results_dir / "ttt_validation_results.json"
        if validation_file.exists():
            with open(validation_file, 'r') as f:
                self.report_data['ttt_validation'] = json.load(f)
        
        # Load error margin analysis
        error_file = self.results_dir / "error_margin_analysis.json"
        if error_file.exists():
            with open(error_file, 'r') as f:
                self.report_data['error_margin_analysis'] = json.load(f)
        
        print(f"Loaded {len(self.report_data)} test result categories")
    
    def _analyze_results(self):
        """Analyze all test results."""
        print("Analyzing results...")
        
        self.analysis = {
            'energy_optimization': self._analyze_energy_optimization(),
            'thermal_management': self._analyze_thermal_management(),
            'noise_robustness': self._analyze_noise_robustness(),
            'routing_efficiency': self._analyze_routing_efficiency(),
            'convergence_analysis': self._analyze_convergence(),
            'overall_performance': self._analyze_overall_performance()
        }
    
    def _analyze_energy_optimization(self) -> Dict[str, Any]:
        """Analyze energy optimization results."""
        analysis = {}
        
        if 'energy_aware_ttt' in self.report_data:
            lambda_values = []
            energy_savings = []
            accuracy_losses = []
            
            for lambda_val, data in self.report_data['energy_aware_ttt'].items():
                lambda_values.append(float(lambda_val))
                energy_savings.append(data['results']['energy_savings_percent'])
                accuracy_losses.append(data['results']['accuracy_loss_percent'])
            
            # Find optimal lambda
            if energy_savings:
                optimal_idx = np.argmax(energy_savings)
                analysis['optimal_lambda'] = lambda_values[optimal_idx]
                analysis['max_energy_savings'] = energy_savings[optimal_idx]
                analysis['corresponding_accuracy_loss'] = accuracy_losses[optimal_idx]
                
                # Calculate energy-accuracy trade-off
                analysis['energy_accuracy_tradeoff'] = {
                    'lambda_values': lambda_values,
                    'energy_savings': energy_savings,
                    'accuracy_losses': accuracy_losses
                }
        
        return analysis
    
    def _analyze_thermal_management(self) -> Dict[str, Any]:
        """Analyze thermal management results."""
        analysis = {}
        
        if 'thermal_aware_routing' in self.report_data:
            thermal_data = self.report_data['thermal_aware_routing']
            
            if 'results' in thermal_data:
                results = thermal_data['results']
                
                # Analyze thermal scenarios
                scenarios = {}
                for result in results:
                    scenario = result['thermal_scenario']
                    if scenario not in scenarios:
                        scenarios[scenario] = {
                            'temperatures': [],
                            'imbalance_scores': [],
                            'migration_counts': [],
                            'energies': []
                        }
                    
                    scenarios[scenario]['temperatures'].append(result['avg_temperature'])
                    scenarios[scenario]['imbalance_scores'].append(result['thermal_imbalance_score'])
                    scenarios[scenario]['migration_counts'].append(result['expert_migration_count'])
                    scenarios[scenario]['energies'].append(result['avg_energy_joules'])
                
                # Calculate averages
                for scenario, data in scenarios.items():
                    scenarios[scenario]['avg_temperature'] = np.mean(data['temperatures'])
                    scenarios[scenario]['avg_imbalance'] = np.mean(data['imbalance_scores'])
                    scenarios[scenario]['avg_migrations'] = np.mean(data['migration_counts'])
                    scenarios[scenario]['avg_energy'] = np.mean(data['energies'])
                
                analysis['thermal_scenarios'] = scenarios
                
                # Find most challenging scenario
                max_imbalance = 0
                worst_scenario = None
                for scenario, data in scenarios.items():
                    if data['avg_imbalance'] > max_imbalance:
                        max_imbalance = data['avg_imbalance']
                        worst_scenario = scenario
                
                analysis['worst_thermal_scenario'] = worst_scenario
                analysis['max_thermal_imbalance'] = max_imbalance
        
        return analysis
    
    def _analyze_noise_robustness(self) -> Dict[str, Any]:
        """Analyze noise robustness results."""
        analysis = {}
        
        if 'error_margin_analysis' in self.report_data:
            error_data = self.report_data['error_margin_analysis']
            
            if 'results' in error_data:
                results = error_data['results']
                
                # Analyze noise levels
                noise_levels = []
                robustness_scores = []
                
                for result in results:
                    noise_levels.append(result.get('noise_level', 0))
                    robustness_scores.append(result.get('robustness_score', 0))
                
                analysis['noise_robustness'] = {
                    'noise_levels': noise_levels,
                    'robustness_scores': robustness_scores
                }
                
                # Find noise tolerance threshold
                if robustness_scores:
                    threshold_idx = None
                    for i, score in enumerate(robustness_scores):
                        if score < 0.8:  # 80% robustness threshold
                            threshold_idx = i
                            break
                    
                    if threshold_idx is not None:
                        analysis['noise_tolerance_threshold'] = noise_levels[threshold_idx]
        
        return analysis
    
    def _analyze_routing_efficiency(self) -> Dict[str, Any]:
        """Analyze routing efficiency results."""
        analysis = {}
        
        if 'ttt_validation' in self.report_data:
            validation_data = self.report_data['ttt_validation']
            
            if 'validation_results' in validation_data:
                results = validation_data['validation_results']
                
                analysis['routing_entropy'] = results.get('routing_entropy', 0)
                analysis['ttt_update_count'] = results.get('ttt_update_count', 0)
                analysis['convergence_rate'] = results.get('convergence_rate', 0)
                
                # Calculate routing efficiency
                if analysis['routing_entropy'] > 0:
                    efficiency = min(1.0, analysis['routing_entropy'] / np.log(16))  # 16 experts
                    analysis['routing_efficiency'] = efficiency
        
        return analysis
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence results."""
        analysis = {}
        
        if 'ttt_validation' in self.report_data:
            validation_data = self.report_data['ttt_validation']
            
            if 'validation_results' in validation_data:
                results = validation_data['validation_results']
                
                analysis['convergence_rate'] = results.get('convergence_rate', 0)
                analysis['ttt_update_count'] = results.get('ttt_update_count', 0)
                
                # Assess convergence quality
                if analysis['convergence_rate'] > 0.8:
                    analysis['convergence_quality'] = 'Excellent'
                elif analysis['convergence_rate'] > 0.6:
                    analysis['convergence_quality'] = 'Good'
                elif analysis['convergence_rate'] > 0.4:
                    analysis['convergence_quality'] = 'Fair'
                else:
                    analysis['convergence_quality'] = 'Poor'
        
        return analysis
    
    def _analyze_overall_performance(self) -> Dict[str, Any]:
        """Analyze overall performance."""
        analysis = {}
        
        # Aggregate performance metrics
        energy_savings = []
        accuracy_losses = []
        thermal_scores = []
        
        # Collect from energy-aware TTT
        if 'energy_aware_ttt' in self.report_data:
            for data in self.report_data['energy_aware_ttt'].values():
                if 'results' in data:
                    energy_savings.append(data['results']['energy_savings_percent'])
                    accuracy_losses.append(data['results']['accuracy_loss_percent'])
        
        # Collect from TTT validation
        if 'ttt_validation' in self.report_data:
            validation_data = self.report_data['ttt_validation']
            if 'validation_results' in validation_data:
                results = validation_data['validation_results']
                thermal_scores.append(results.get('thermal_adaptation_score', 0))
        
        # Calculate overall metrics
        if energy_savings:
            analysis['avg_energy_savings'] = np.mean(energy_savings)
            analysis['max_energy_savings'] = np.max(energy_savings)
            analysis['avg_accuracy_loss'] = np.mean(accuracy_losses)
        
        if thermal_scores:
            analysis['avg_thermal_adaptation'] = np.mean(thermal_scores)
        
        # Overall performance score
        performance_score = 0
        if energy_savings:
            performance_score += np.mean(energy_savings) * 0.4  # 40% weight
        if accuracy_losses:
            performance_score += (1 - np.mean(accuracy_losses) / 100) * 0.3  # 30% weight
        if thermal_scores:
            performance_score += np.mean(thermal_scores) * 0.3  # 30% weight
        
        analysis['overall_performance_score'] = performance_score
        
        # Performance grade
        if performance_score > 0.8:
            analysis['performance_grade'] = 'A'
        elif performance_score > 0.7:
            analysis['performance_grade'] = 'B'
        elif performance_score > 0.6:
            analysis['performance_grade'] = 'C'
        else:
            analysis['performance_grade'] = 'D'
        
        return analysis
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create comprehensive summary."""
        summary = {
            'test_overview': {
                'total_tests': len(self.report_data),
                'test_categories': list(self.report_data.keys()),
                'timestamp': self.analysis.get('overall_performance', {}).get('timestamp', 0)
            },
            'key_findings': self._extract_key_findings(),
            'performance_summary': self.analysis['overall_performance'],
            'energy_optimization': self.analysis['energy_optimization'],
            'thermal_management': self.analysis['thermal_management'],
            'noise_robustness': self.analysis['noise_robustness'],
            'routing_efficiency': self.analysis['routing_efficiency'],
            'convergence_analysis': self.analysis['convergence_analysis'],
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from analysis."""
        findings = []
        
        # Energy optimization findings
        energy_analysis = self.analysis['energy_optimization']
        if 'optimal_lambda' in energy_analysis:
            findings.append(f"Optimal energy penalty (lambda): {energy_analysis['optimal_lambda']:.3f}")
            findings.append(f"Maximum energy savings: {energy_analysis['max_energy_savings']:.2f}%")
        
        # Thermal management findings
        thermal_analysis = self.analysis['thermal_management']
        if 'worst_thermal_scenario' in thermal_analysis:
            findings.append(f"Most challenging thermal scenario: {thermal_analysis['worst_thermal_scenario']}")
        
        # Overall performance findings
        perf_analysis = self.analysis['overall_performance']
        if 'performance_grade' in perf_analysis:
            findings.append(f"Overall performance grade: {perf_analysis['performance_grade']}")
        
        # Convergence findings
        conv_analysis = self.analysis['convergence_analysis']
        if 'convergence_quality' in conv_analysis:
            findings.append(f"TTT convergence quality: {conv_analysis['convergence_quality']}")
        
        return findings
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Energy optimization recommendations
        energy_analysis = self.analysis['energy_optimization']
        if 'optimal_lambda' in energy_analysis:
            recommendations.append(f"Use lambda_energy={energy_analysis['optimal_lambda']:.3f} for optimal energy-accuracy trade-off")
        
        # Thermal management recommendations
        thermal_analysis = self.analysis['thermal_management']
        if 'max_thermal_imbalance' in thermal_analysis:
            if thermal_analysis['max_thermal_imbalance'] > 0.3:
                recommendations.append("Consider implementing more aggressive thermal management strategies")
            else:
                recommendations.append("Current thermal management is effective")
        
        # Noise robustness recommendations
        noise_analysis = self.analysis['noise_robustness']
        if 'noise_tolerance_threshold' in noise_analysis:
            recommendations.append(f"System noise tolerance threshold: {noise_analysis['noise_tolerance_threshold']:.3f}")
        
        # Overall recommendations
        perf_analysis = self.analysis['overall_performance']
        if 'performance_grade' in perf_analysis:
            if perf_analysis['performance_grade'] in ['A', 'B']:
                recommendations.append("System is ready for production deployment")
            else:
                recommendations.append("System needs further optimization before production")
        
        return recommendations
    
    def _generate_plots(self):
        """Generate summary plots."""
        print("Generating summary plots...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Synthetic TTT Test Results Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Energy-Accuracy Trade-off
        if 'energy_optimization' in self.analysis:
            energy_data = self.analysis['energy_optimization']
            if 'energy_accuracy_tradeoff' in energy_data:
                tradeoff = energy_data['energy_accuracy_tradeoff']
                axes[0, 0].plot(tradeoff['lambda_values'], tradeoff['energy_savings'], 'bo-', label='Energy Savings')
                axes[0, 0].set_xlabel('Lambda Energy')
                axes[0, 0].set_ylabel('Energy Savings (%)')
                axes[0, 0].set_title('Energy Optimization')
                axes[0, 0].grid(True)
        
        # Plot 2: Thermal Scenarios
        if 'thermal_management' in self.analysis:
            thermal_data = self.analysis['thermal_management']
            if 'thermal_scenarios' in thermal_data:
                scenarios = thermal_data['thermal_scenarios']
                scenario_names = list(scenarios.keys())
                avg_temps = [scenarios[s]['avg_temperature'] for s in scenario_names]
                avg_imbalances = [scenarios[s]['avg_imbalance'] for s in scenario_names]
                
                x = np.arange(len(scenario_names))
                width = 0.35
                
                axes[0, 1].bar(x - width/2, avg_temps, width, label='Avg Temperature (°C)')
                axes[0, 1].bar(x + width/2, [i * 100 for i in avg_imbalances], width, label='Thermal Imbalance (%)')
                axes[0, 1].set_xlabel('Thermal Scenario')
                axes[0, 1].set_ylabel('Temperature (°C) / Imbalance (%)')
                axes[0, 1].set_title('Thermal Management')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(scenario_names)
                axes[0, 1].legend()
                axes[0, 1].grid(True)
        
        # Plot 3: Noise Robustness
        if 'noise_robustness' in self.analysis:
            noise_data = self.analysis['noise_robustness']
            if 'noise_robustness' in noise_data:
                robustness = noise_data['noise_robustness']
                axes[0, 2].plot(robustness['noise_levels'], robustness['robustness_scores'], 'go-')
                axes[0, 2].set_xlabel('Noise Level')
                axes[0, 2].set_ylabel('Robustness Score')
                axes[0, 2].set_title('Noise Robustness')
                axes[0, 2].grid(True)
        
        # Plot 4: Routing Efficiency
        if 'routing_efficiency' in self.analysis:
            routing_data = self.analysis['routing_efficiency']
            metrics = ['Routing Entropy', 'TTT Updates', 'Convergence Rate']
            values = [
                routing_data.get('routing_entropy', 0),
                routing_data.get('ttt_update_count', 0) / 1000,  # Normalize
                routing_data.get('convergence_rate', 0)
            ]
            
            axes[1, 0].bar(metrics, values, color=['skyblue', 'lightgreen', 'orange'])
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Routing Efficiency')
            axes[1, 0].grid(True)
        
        # Plot 5: Overall Performance
        if 'overall_performance' in self.analysis:
            perf_data = self.analysis['overall_performance']
            metrics = ['Energy Savings', 'Accuracy Loss', 'Thermal Adaptation']
            values = [
                perf_data.get('avg_energy_savings', 0),
                100 - perf_data.get('avg_accuracy_loss', 0),  # Convert to accuracy
                perf_data.get('avg_thermal_adaptation', 0) * 100
            ]
            
            axes[1, 1].bar(metrics, values, color=['red', 'blue', 'green'])
            axes[1, 1].set_ylabel('Score (%)')
            axes[1, 1].set_title('Overall Performance')
            axes[1, 1].grid(True)
        
        # Plot 6: Performance Grade
        if 'overall_performance' in self.analysis:
            perf_data = self.analysis['overall_performance']
            if 'performance_grade' in perf_data:
                grade = perf_data['performance_grade']
                score = perf_data.get('overall_performance_score', 0)
                
                axes[1, 2].text(0.5, 0.5, f'Grade: {grade}\nScore: {score:.3f}', 
                               ha='center', va='center', fontsize=24, fontweight='bold')
                axes[1, 2].set_xlim(0, 1)
                axes[1, 2].set_ylim(0, 1)
                axes[1, 2].set_title('Performance Grade')
                axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / "synthetic_test_summary_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Summary plots saved to {plot_file}")
        
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Test Summary Report")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--output_file", type=str, required=True, help="Output summary file")
    parser.add_argument("--generate_plots", action="store_true", help="Generate summary plots")
    
    args = parser.parse_args()
    
    print("=== Synthetic Test Summary Report Generator ===")
    print(f"Results Directory: {args.results_dir}")
    print(f"Output File: {args.output_file}")
    print(f"Generate Plots: {args.generate_plots}")
    
    # Generate report
    generator = SyntheticTestReportGenerator(args.results_dir)
    generator.generate_report(args.output_file, args.generate_plots)
    
    print("\n=== Report Generation Complete ===")

if __name__ == "__main__":
    main() 