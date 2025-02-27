# plotting.py
"""
Satellite System Analysis Visualization Module
=========================================

This module provides visualization tools for analyzing satellite system performance
across different configurations and strategies.

Plot Types
---------
1. Cost vs. MAU Analysis:
   - Scatter plot showing relationship between total cost and MAU scores
   - Helps identify cost-effective configurations

2. Tradespace Analysis:
   - Multi-dimensional visualization of:
     * Cost ($)
     * MAU score
     * Option type
     * Urgency level
     * Capacity utilization

3. MAU S-Curve:
   - Cumulative probability distribution of MAU scores
   - Shows risk and uncertainty across options

4. Coverage & Redundancy:
   - Dual plot showing:
     * Coverage area distribution
     * System redundancy levels
   - Box plots for statistical distribution

5. Sensitivity Analysis:
   - Tornado diagram showing parameter impacts
   - Grouped by parameter categories:
     * Integration Parameters
     * Coverage & Performance
     * Satellite Availability
     * Regional Distribution
     * Other Parameters

Memory Management
---------------
- Uses plt.close() to manage figure resources
- Implements optional figure saving
- Efficient handling of large datasets

Style Configuration
-----------------
- Consistent styling across all plots
- Configurable through PlotConfig class
- Default to publication-quality settings
"""

from typing import Optional, List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class PlotConfig:
    """
    Configuration for plot styling and formatting.
    
    Attributes:
        figsize: Default figure size (width, height)
        style: matplotlib/seaborn style
        palette: Color palette name
        dpi: Resolution in dots per inch
        format: Output file format
        title_fontsize: Title font size
        label_fontsize: Axis label font size
        legend_fontsize: Legend font size
        grid: Show grid lines
    """
    figsize: Tuple[int, int] = (12, 8)
    style: str = 'seaborn'
    palette: str = 'tab10'
    dpi: int = 300
    format: str = 'png'
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 10
    grid: bool = True

class PlotManager:
    """
    Handles creation and management of visualization plots with consistent styling.
    
    Features:
    - Consistent styling across plots
    - Resource management
    - Optional plot saving
    - Error handling and logging
    """

    def __init__(
        self,
        config: PlotConfig = PlotConfig(),
        save_dir: Optional[str] = None
    ):
        """Initialize plot manager with configuration."""
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else None
        plt.style.use(config.style)
        sns.set_palette(config.palette)

    def setup_figure(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        subplot_kw: Optional[Dict] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create and configure a new figure with proper resource management.
        
        Args:
            figsize: Optional custom figure size
            subplot_kw: Optional subplot parameters
            
        Returns:
            Tuple of (Figure, Axes)
        """
        fig, ax = plt.subplots(
            figsize=figsize or self.config.figsize,
            dpi=self.config.dpi,
            subplot_kw=subplot_kw or {}
        )
        if self.config.grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        return fig, ax

    def save_plot(
        self,
        fig: plt.Figure,
        name: str,
        close: bool = True
    ) -> None:
        """
        Save plot with proper resource cleanup.
        
        Args:
            fig: Figure to save
            name: Filename without extension
            close: Whether to close figure after saving
        """
        try:
            if self.save_dir:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                filepath = self.save_dir / f"{name}.{self.config.format}"
                fig.savefig(
                    filepath,
                    bbox_inches='tight',
                    dpi=self.config.dpi
                )
                logger.info(f"Saved plot to {filepath}")
        except Exception as e:
            logger.error(f"Error saving plot {name}: {str(e)}")
        finally:
            if close:
                plt.close(fig)

    def plot_cost_vs_mau(
        self,
        results_df: pd.DataFrame,
        show: bool = True
    ) -> None:
        """
        Create cost vs MAU scatter plot.
        
        Visualizes cost-effectiveness by plotting:
        - Total Cost ($) on x-axis
        - MAU score on y-axis
        - Different options as distinct markers
        """
        try:
            fig, ax = self.setup_figure()
            
            sns.scatterplot(
                data=results_df,
                x='Total Cost ($)',
                y='MAU',
                hue='Option',
                style='Option',
                s=100,
                ax=ax
            )

            self._format_axes(
                ax,
                xlabel='Total Cost ($)',
                ylabel='MAU',
                title='Cost-Effectiveness Analysis'
            )
            
            ax.legend(
                title='Option',
                fontsize=self.config.legend_fontsize,
                title_fontsize=self.config.legend_fontsize
            )

            plt.tight_layout()
            self.save_plot(fig, 'cost_vs_mau')
            
            if show:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error in cost vs MAU plot: {str(e)}")
            raise

    def plot_tradespace(
        self,
        results_df: pd.DataFrame,
        show: bool = True
    ) -> None:
        """
        Create multi-dimensional tradespace visualization.
        
        Dimensions:
        - Cost ($) on x-axis
        - MAU score on y-axis
        - Option type as color
        - Urgency as marker style
        - Percentage use as marker size
        """
        try:
            fig, ax = self.setup_figure()
            
            scatter = sns.scatterplot(
                data=results_df,
                x='Total Cost ($)',
                y='MAU',
                hue='Option',
                style='Urgency',
                size='Percentage Use',
                sizes=(50, 200),
                alpha=0.7,
                ax=ax
            )

            self._format_axes(
                ax,
                xlabel='Total Cost ($)',
                ylabel='MAU',
                title='Multi-Dimensional Tradespace Analysis'
            )

            scatter.legend(
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                fontsize=self.config.legend_fontsize
            )

            plt.tight_layout()
            self.save_plot(fig, 'tradespace_analysis')
            
            if show:
                plt.show()

        except Exception as e:
            logger.error(f"Error in tradespace plot: {str(e)}")
            raise

    def plot_mau_s_curve(
        self,
        results_df: pd.DataFrame,
        show: bool = True
    ) -> None:
        """
        Create MAU probability S-curve plot.
        
        Shows:
        - Cumulative probability distribution
        - Risk profile for each option
        - Uncertainty comparison
        """
        try:
            fig, ax = self.setup_figure()
            
            # Calculate curves for each option
            for option in results_df['Option'].unique():
                option_data = results_df[results_df['Option'] == option]
                maus = np.sort(option_data['MAU'])
                cumulative_prob = np.arange(1, len(maus) + 1) / len(maus)
                ax.plot(maus, cumulative_prob, label=option, linewidth=2)

            self._format_axes(
                ax,
                xlabel='MAU Score',
                ylabel='Cumulative Probability',
                title='MAU Probability Distribution'
            )
            
            ax.legend(
                title='Option',
                fontsize=self.config.legend_fontsize,
                title_fontsize=self.config.legend_fontsize
            )

            plt.tight_layout()
            self.save_plot(fig, 'mau_s_curve')
            
            if show:
                plt.show()

        except Exception as e:
            logger.error(f"Error in MAU S-curve plot: {str(e)}")
            raise

    def plot_coverage_and_redundancy(
        self,
        results_df: pd.DataFrame,
        show: bool = True
    ) -> None:
        """
        Create dual plot of coverage and redundancy metrics.
        
        Shows:
        - Coverage area distribution by option
        - Redundancy levels by option
        - Statistical distribution via box plots
        """
        try:
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(self.config.figsize[0], self.config.figsize[1] * 1.2),
                dpi=self.config.dpi
            )

            # Coverage plot
            sns.boxplot(
                data=results_df,
                x='Option',
                y='Coverage Area (sq km)',
                ax=ax1
            )
            self._format_axes(
                ax1,
                xlabel='',
                ylabel='Coverage Area (sq km)',
                title='Coverage Distribution by Option'
            )

            # Redundancy plot
            sns.boxplot(
                data=results_df,
                x='Option',
                y='Redundancy',
                ax=ax2
            )
            self._format_axes(
                ax2,
                xlabel='Option',
                ylabel='Redundancy',
                title='Redundancy Distribution by Option'
            )

            plt.tight_layout()
            self.save_plot(fig, 'coverage_redundancy')
            
            if show:
                plt.show()

        except Exception as e:
            logger.error(f"Error in coverage/redundancy plot: {str(e)}")
            raise

    def plot_tornado_diagram(
        self,
        sensitivity_results: pd.DataFrame,
        show: bool = True
    ) -> None:
        """
        Create enhanced tornado diagram for sensitivity analysis.
        
        Features:
        - Parameter grouping by category
        - Color-coded bars
        - Category labels
        - Organized layout
        """
        try:
            # Rename columns if needed
            if 'High Impact' in sensitivity_results.columns:
                sensitivity_results = sensitivity_results.rename(
                    columns={
                        'High Impact': 'High_Impact',
                        'Low Impact': 'Low_Impact'
                    }
                )

            # Group variables by category
            categories = {
                'Integration Parameters': [
                    'Integration Time - Non-Integrated',
                    'Integration Batch Size',
                    'Standard Integration Time',
                    'Accelerated Integration Time',
                    'Critical Integration Time'
                ],
                'Coverage & Performance': [
                    'Coverage Radius Multiplier',
                    'Redundancy Threshold',
                    'Percentage Use'
                ],
                'Satellite Availability': [
                    'MILSATCOM Availability Probability',
                    'COMSATCOM Availability Probability'
                ],
                'Regional Distribution': [
                    'Regional Weight - CONUS',
                    'Regional Weight - APAC',
                    'Regional Weight - MENA',
                    'Regional Weight - Europe',
                    'Regional Weight - Africa',
                    'Regional Weight - LATAM',
                    'Regional Weight - Oceania'
                ],
                'Other Parameters': [
                    'MILSATCOM/COMSATCOM Split',
                    'Demand Volatility'
                ]
            }

            # Create figure
            fig, ax = plt.subplots(
                figsize=(self.config.figsize[0], self.config.figsize[1] * 1.5),
                dpi=self.config.dpi
            )

            # Process and sort data
            plot_data = []
            y_labels = []
            category_positions = {}
            current_position = 0

            # Process each category
            for category, vars_in_category in categories.items():
                category_data = sensitivity_results[
                    sensitivity_results['Variable'].isin(vars_in_category)
                ]
                
                if not category_data.empty:
                    # Sort by impact range within category
                    category_data = category_data.sort_values('Impact Range', ascending=True)
                    
                    # Store category midpoint
                    category_positions[category] = current_position + len(category_data) / 2
                    
                    # Add each variable in category
                    for _, row in category_data.iterrows():
                        plot_data.append({
                            'position': current_position,
                            'variable': row['Variable'],
                            'high_impact': row['High_Impact'],
                            'low_impact': row['Low_Impact']
                        })
                        y_labels.append(row['Variable'])
                        current_position += 1
                    
                    # Add space between categories
                    current_position += 0.5

            # Create y-positions array
            y_positions = [d['position'] for d in plot_data]

            # Plot bars
            colors = ['skyblue' if i % 2 == 0 else 'lightblue' for i in range(len(plot_data))]
            
            for i, data in enumerate(plot_data):
                width = data['high_impact'] - data['low_impact']
                center = (data['high_impact'] + data['low_impact']) / 2
                ax.barh(
                    data['position'],
                    width,
                    left=center - width/2,
                    height=0.8,
                    color=colors[i],
                    alpha=0.7
                )

            # Add category labels
            for category, position in category_positions.items():
                ax.text(
                    ax.get_xlim()[1],  # Right side of plot
                    position,
                    f' {category}',
                    fontsize=self.config.label_fontsize,
                    fontweight='bold',
                    va='center'
                )

            # Customize plot
            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels, fontsize=self.config.label_fontsize)
            ax.set_xlabel(
                'Impact on MAU (%)',
                fontsize=self.config.label_fontsize
            )
            ax.set_title(
                'Sensitivity Analysis: Operational Parameters',
                fontsize=self.config.title_fontsize
            )

            # Add zero line
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

            # Add grid
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            
            self.save_plot(fig, 'tornado_diagram')
            
            if show:
                plt.show()

        except Exception as e:
            logger.error(f"Error in tornado diagram: {str(e)}")
            raise

    def _format_axes(
    self,
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str
) -> None:
        """Helper method for consistent axes formatting."""
        ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.tick_params(labelsize=self.config.label_fontsize)

