"""
Language Learning Engagement Analyzer
A product analytics framework for understanding learner engagement patterns.

Author: Kajal Patel
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import scipy.stats as stats


class StreakAnalyzer:
    """
    Analyzes streak behavior to identify patterns leading to streak breaks.
    
    Key insights this provides:
    - What time of day do users typically break streaks?
    - Which lessons have highest streak-break correlation?
    - Who are at-risk users for churn?
    """
    
    def __init__(self, user_activity_df: pd.DataFrame):
        """
        Initialize with user activity data.
        
        Args:
            user_activity_df: DataFrame with columns:
                - user_id: unique user identifier
                - activity_date: date of activity
                - streak_days: current streak length
                - last_activity_date: most recent activity
                - lesson_id: lesson completed (if any)
        """
        self.data = user_activity_df.copy()
        self._preprocess_dates()
    
    def _preprocess_dates(self):
        """Convert date columns to datetime."""
        date_cols = ['activity_date', 'last_activity_date']
        for col in date_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col])
    
    def calculate_streak_metrics(self) -> Dict[str, float]:
        """
        Calculate key streak engagement metrics.
        
        Returns:
            Dictionary containing:
            - avg_streak_length: Mean streak length across all users
            - median_streak_length: Median streak length
            - streak_break_rate: Rate of streak breaks
            - pct_users_with_7day_streak: % of users with 7+ day streaks
        """
        user_streaks = self.data.groupby('user_id')['streak_days'].max()
        
        metrics = {
            'avg_streak_length': user_streaks.mean(),
            'median_streak_length': user_streaks.median(),
            'max_streak_length': user_streaks.max(),
            'pct_users_with_7day_streak': (user_streaks >= 7).mean() * 100,
            'pct_users_with_30day_streak': (user_streaks >= 30).mean() * 100,
            'pct_users_with_100day_streak': (user_streaks >= 100).mean() * 100,
        }
        return metrics
    
    def identify_at_risk_users(self, threshold_days: int = 2) -> pd.DataFrame:
        """
        Identify users who haven't practiced recently.
        These are candidates for re-engagement notifications.
        
        Args:
            threshold_days: Days of inactivity to flag as at-risk
            
        Returns:
            DataFrame of at-risk users with their streak info
        """
        today = datetime.now()
        
        # Get most recent activity per user
        latest_activity = self.data.groupby('user_id').agg({
            'last_activity_date': 'max',
            'streak_days': 'last'
        }).reset_index()
        
        # Calculate days since last activity
        latest_activity['days_inactive'] = (
            today - latest_activity['last_activity_date']
        ).dt.days
        
        # Filter to at-risk users
        at_risk = latest_activity[
            latest_activity['days_inactive'] >= threshold_days
        ].copy()
        
        # Add risk level
        at_risk['risk_level'] = pd.cut(
            at_risk['days_inactive'],
            bins=[0, 2, 5, 14, float('inf')],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return at_risk.sort_values('streak_days', ascending=False)
    
    def streak_break_patterns(self) -> Dict[str, pd.Series]:
        """
        Analyze when and why streaks break.
        
        Returns:
            Dictionary with break patterns by various dimensions
        """
        # Identify streak breaks (days where streak reset to 0 or 1)
        streak_breaks = self.data[self.data['streak_days'] <= 1].copy()
        
        if streak_breaks.empty:
            return {'message': 'No streak breaks found in data'}
        
        patterns = {}
        
        # Breaks by day of week
        if 'activity_date' in streak_breaks.columns:
            streak_breaks['day_of_week'] = streak_breaks['activity_date'].dt.day_name()
            patterns['by_day_of_week'] = streak_breaks['day_of_week'].value_counts()
        
        # Breaks by hour (if available)
        if 'activity_hour' in streak_breaks.columns:
            patterns['by_hour'] = streak_breaks['activity_hour'].value_counts().sort_index()
        
        # Breaks by lesson type (if available)
        if 'lesson_type' in streak_breaks.columns:
            patterns['by_lesson_type'] = streak_breaks['lesson_type'].value_counts()
        
        return patterns
    
    def cohort_retention_analysis(self, cohort_col: str = 'signup_month') -> pd.DataFrame:
        """
        Analyze streak retention by user cohort.
        
        Args:
            cohort_col: Column to use for cohort grouping
            
        Returns:
            DataFrame with retention rates by cohort over time
        """
        if cohort_col not in self.data.columns:
            raise ValueError(f"Column {cohort_col} not found in data")
        
        retention = self.data.groupby([cohort_col, 'weeks_since_signup']).agg({
            'user_id': 'nunique'
        }).reset_index()
        
        # Calculate retention rate relative to week 0
        week_0_users = retention[retention['weeks_since_signup'] == 0].set_index(cohort_col)['user_id']
        
        retention['retention_rate'] = retention.apply(
            lambda row: row['user_id'] / week_0_users.get(row[cohort_col], 1) * 100,
            axis=1
        )
        
        return retention.pivot(
            index='weeks_since_signup',
            columns=cohort_col,
            values='retention_rate'
        )


class LearningFrictionAnalyzer:
    """
    Identifies points in the learning journey where users drop off.
    Helps PMs prioritize UX improvements and content optimization.
    """
    
    def __init__(self, lesson_data: pd.DataFrame, user_journeys: pd.DataFrame):
        """
        Initialize with lesson and journey data.
        
        Args:
            lesson_data: DataFrame with lesson metadata (lesson_id, skill_type, difficulty, etc.)
            user_journeys: DataFrame with user lesson interactions (user_id, lesson_id, started, completed, time_spent)
        """
        self.lessons = lesson_data.copy()
        self.journeys = user_journeys.copy()
    
    def identify_drop_off_points(self, top_n: int = 10) -> pd.DataFrame:
        """
        Find lessons with highest abandonment rates.
        
        Args:
            top_n: Number of high-friction lessons to return
            
        Returns:
            DataFrame of lessons ranked by drop-off rate
        """
        drop_offs = self.journeys.groupby('lesson_id').agg({
            'started': 'sum',
            'completed': 'sum'
        }).reset_index()
        
        drop_offs['completion_rate'] = drop_offs['completed'] / drop_offs['started']
        drop_offs['drop_off_rate'] = 1 - drop_offs['completion_rate']
        drop_offs['drop_off_count'] = drop_offs['started'] - drop_offs['completed']
        
        # Merge with lesson metadata
        if 'lesson_id' in self.lessons.columns:
            drop_offs = drop_offs.merge(
                self.lessons[['lesson_id', 'lesson_name', 'skill_type', 'difficulty']],
                on='lesson_id',
                how='left'
            )
        
        return drop_offs.nlargest(top_n, 'drop_off_rate')
    
    def friction_by_skill_type(self) -> pd.DataFrame:
        """
        Analyze friction patterns by skill type (speaking, listening, reading, writing).
        
        Returns:
            DataFrame with completion rates by skill type
        """
        merged = self.journeys.merge(
            self.lessons[['lesson_id', 'skill_type']],
            on='lesson_id',
            how='left'
        )
        
        analysis = merged.groupby('skill_type').agg({
            'started': 'sum',
            'completed': 'sum',
            'time_spent': 'mean'
        }).reset_index()
        
        analysis['completion_rate'] = analysis['completed'] / analysis['started'] * 100
        analysis['avg_time_minutes'] = analysis['time_spent'] / 60
        
        return analysis.sort_values('completion_rate', ascending=False)
    
    def time_to_completion_analysis(self) -> pd.DataFrame:
        """
        Analyze how long lessons take vs. expected time.
        Lessons taking too long may indicate difficulty issues.
        
        Returns:
            DataFrame with time analysis by lesson
        """
        if 'expected_time' not in self.journeys.columns:
            raise ValueError("expected_time column required for this analysis")
        
        time_analysis = self.journeys.copy()
        time_analysis['time_ratio'] = (
            time_analysis['time_spent'] / time_analysis['expected_time']
        )
        
        summary = time_analysis.groupby('lesson_id').agg({
            'time_ratio': ['mean', 'median', 'std'],
            'time_spent': 'mean'
        }).reset_index()
        
        summary.columns = ['lesson_id', 'avg_time_ratio', 'median_time_ratio', 
                          'time_ratio_std', 'avg_time_spent']
        
        # Flag lessons that take significantly longer than expected
        summary['needs_review'] = summary['avg_time_ratio'] > 1.5
        
        return summary.sort_values('avg_time_ratio', ascending=False)
    
    def user_journey_funnel(self, skill_sequence: List[str]) -> pd.DataFrame:
        """
        Analyze user progression through a skill sequence.
        
        Args:
            skill_sequence: Ordered list of skills in the learning path
            
        Returns:
            DataFrame showing funnel conversion at each step
        """
        funnel_data = []
        
        for i, skill in enumerate(skill_sequence):
            users_reached = self.journeys[
                self.journeys['skill_type'] == skill
            ]['user_id'].nunique()
            
            users_completed = self.journeys[
                (self.journeys['skill_type'] == skill) & 
                (self.journeys['completed'] == True)
            ]['user_id'].nunique()
            
            funnel_data.append({
                'step': i + 1,
                'skill': skill,
                'users_reached': users_reached,
                'users_completed': users_completed,
                'step_completion_rate': users_completed / users_reached * 100 if users_reached > 0 else 0
            })
        
        funnel_df = pd.DataFrame(funnel_data)
        
        # Calculate cumulative conversion
        initial_users = funnel_df.iloc[0]['users_reached']
        funnel_df['cumulative_conversion'] = funnel_df['users_completed'] / initial_users * 100
        
        return funnel_df


class ABTestAnalyzer:
    """
    Framework for analyzing A/B test results on engagement features.
    Supports product decisions on gamification, notifications, and UX changes.
    """
    
    def __init__(self, control_group: pd.DataFrame, treatment_group: pd.DataFrame):
        """
        Initialize with test group data.
        
        Args:
            control_group: DataFrame with control group user data
            treatment_group: DataFrame with treatment group user data
        """
        self.control = control_group.copy()
        self.treatment = treatment_group.copy()
    
    def calculate_engagement_lift(self, metric: str = 'daily_sessions') -> Dict:
        """
        Calculate lift in engagement metric.
        
        Args:
            metric: Column name for the metric to compare
            
        Returns:
            Dictionary with lift analysis results
        """
        if metric not in self.control.columns or metric not in self.treatment.columns:
            raise ValueError(f"Metric '{metric}' not found in both groups")
        
        control_values = self.control[metric].dropna()
        treatment_values = self.treatment[metric].dropna()
        
        control_mean = control_values.mean()
        treatment_mean = treatment_values.mean()
        
        lift = (treatment_mean - control_mean) / control_mean * 100 if control_mean != 0 else 0
        
        # Statistical significance test (two-sample t-test)
        t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
        
        # Calculate confidence interval for the difference
        se_diff = np.sqrt(
            control_values.var() / len(control_values) + 
            treatment_values.var() / len(treatment_values)
        )
        ci_95 = (
            (treatment_mean - control_mean) - 1.96 * se_diff,
            (treatment_mean - control_mean) + 1.96 * se_diff
        )
        
        return {
            'metric': metric,
            'control_mean': round(control_mean, 4),
            'treatment_mean': round(treatment_mean, 4),
            'absolute_difference': round(treatment_mean - control_mean, 4),
            'lift_percent': round(lift, 2),
            't_statistic': round(t_stat, 4),
            'p_value': round(p_value, 4),
            'significant_at_95': p_value < 0.05,
            'significant_at_99': p_value < 0.01,
            'confidence_interval_95': (round(ci_95[0], 4), round(ci_95[1], 4)),
            'control_n': len(control_values),
            'treatment_n': len(treatment_values)
        }
    
    def retention_curve_comparison(self, days: int = 30) -> pd.DataFrame:
        """
        Compare retention curves between test groups.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            DataFrame with retention comparison over time
        """
        if 'days_active' not in self.control.columns:
            raise ValueError("'days_active' column required for retention analysis")
        
        retention_data = []
        
        for day in range(1, days + 1):
            control_retained = (self.control['days_active'] >= day).mean() * 100
            treatment_retained = (self.treatment['days_active'] >= day).mean() * 100
            
            lift = (treatment_retained - control_retained) / control_retained * 100 if control_retained > 0 else 0
            
            retention_data.append({
                'day': day,
                'control_retention': round(control_retained, 2),
                'treatment_retention': round(treatment_retained, 2),
                'retention_lift_pct': round(lift, 2)
            })
        
        return pd.DataFrame(retention_data)
    
    def segment_analysis(self, segment_col: str, metric: str = 'daily_sessions') -> pd.DataFrame:
        """
        Analyze treatment effect across user segments.
        
        Args:
            segment_col: Column to segment by (e.g., 'user_type', 'country')
            metric: Metric to compare
            
        Returns:
            DataFrame with segment-level analysis
        """
        combined = pd.concat([
            self.control.assign(group='control'),
            self.treatment.assign(group='treatment')
        ])
        
        segments = combined[segment_col].unique()
        results = []
        
        for segment in segments:
            control_seg = self.control[self.control[segment_col] == segment]
            treatment_seg = self.treatment[self.treatment[segment_col] == segment]
            
            if len(control_seg) < 30 or len(treatment_seg) < 30:
                continue  # Skip segments with insufficient sample size
            
            control_mean = control_seg[metric].mean()
            treatment_mean = treatment_seg[metric].mean()
            lift = (treatment_mean - control_mean) / control_mean * 100 if control_mean != 0 else 0
            
            _, p_value = stats.ttest_ind(control_seg[metric], treatment_seg[metric])
            
            results.append({
                'segment': segment,
                'control_mean': round(control_mean, 4),
                'treatment_mean': round(treatment_mean, 4),
                'lift_pct': round(lift, 2),
                'p_value': round(p_value, 4),
                'significant': p_value < 0.05,
                'control_n': len(control_seg),
                'treatment_n': len(treatment_seg)
            })
        
        return pd.DataFrame(results).sort_values('lift_pct', ascending=False)
    
    def calculate_sample_size_needed(
        self, 
        baseline_rate: float, 
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        Calculate required sample size for future tests.
        
        Args:
            baseline_rate: Current conversion/engagement rate
            minimum_detectable_effect: Minimum % lift to detect
            alpha: Significance level (default 0.05)
            power: Statistical power (default 0.8)
            
        Returns:
            Required sample size per group
        """
        from scipy.stats import norm
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect / 100)
        
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        
        pooled_p = (p1 + p2) / 2
        
        n = (
            2 * pooled_p * (1 - pooled_p) * (z_alpha + z_beta) ** 2
        ) / (p2 - p1) ** 2
        
        return int(np.ceil(n))


def generate_sample_data(n_users: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate sample data for testing the analyzers.
    
    Args:
        n_users: Number of users to generate
        
    Returns:
        Tuple of (user_activity, lesson_data, user_journeys) DataFrames
    """
    np.random.seed(42)
    
    # User activity data
    user_ids = [f"user_{i}" for i in range(n_users)]
    
    user_activity = pd.DataFrame({
        'user_id': np.repeat(user_ids, 30),  # 30 days of data per user
        'activity_date': pd.date_range(start='2024-01-01', periods=30).tolist() * n_users,
        'streak_days': np.random.geometric(p=0.1, size=n_users * 30),
        'last_activity_date': pd.Timestamp.now() - pd.to_timedelta(
            np.random.exponential(3, n_users * 30), unit='D'
        ),
        'lesson_type': np.random.choice(
            ['speaking', 'listening', 'reading', 'writing'], 
            n_users * 30
        )
    })
    
    # Lesson data
    lesson_ids = [f"lesson_{i}" for i in range(50)]
    lesson_data = pd.DataFrame({
        'lesson_id': lesson_ids,
        'lesson_name': [f"Lesson {i}" for i in range(50)],
        'skill_type': np.random.choice(
            ['speaking', 'listening', 'reading', 'writing'], 50
        ),
        'difficulty': np.random.choice(['easy', 'medium', 'hard'], 50)
    })
    
    # User journeys
    user_journeys = pd.DataFrame({
        'user_id': np.random.choice(user_ids, 5000),
        'lesson_id': np.random.choice(lesson_ids, 5000),
        'started': True,
        'completed': np.random.choice([True, False], 5000, p=[0.75, 0.25]),
        'time_spent': np.random.exponential(300, 5000),  # seconds
        'expected_time': 300
    })
    user_journeys = user_journeys.merge(
        lesson_data[['lesson_id', 'skill_type']], on='lesson_id'
    )
    
    return user_activity, lesson_data, user_journeys


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("Language Learning Engagement Analyzer - Demo")
    print("=" * 60)
    
    # Generate sample data
    user_activity, lesson_data, user_journeys = generate_sample_data(500)
    
    # Streak Analysis
    print("\n📊 STREAK ANALYSIS")
    print("-" * 40)
    streak_analyzer = StreakAnalyzer(user_activity)
    metrics = streak_analyzer.calculate_streak_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    
    # At-risk users
    at_risk = streak_analyzer.identify_at_risk_users(threshold_days=2)
    print(f"\n  At-risk users (2+ days inactive): {len(at_risk)}")
    
    # Friction Analysis
    print("\n📉 LEARNING FRICTION ANALYSIS")
    print("-" * 40)
    friction_analyzer = LearningFrictionAnalyzer(lesson_data, user_journeys)
    
    friction_by_skill = friction_analyzer.friction_by_skill_type()
    print("  Completion rates by skill type:")
    for _, row in friction_by_skill.iterrows():
        print(f"    {row['skill_type']}: {row['completion_rate']:.1f}%")
    
    # A/B Test Analysis
    print("\n🔬 A/B TEST ANALYSIS (Simulated)")
    print("-" * 40)
    
    # Simulate test groups
    np.random.seed(42)
    control = pd.DataFrame({
        'user_id': range(500),
        'daily_sessions': np.random.poisson(2.5, 500),
        'days_active': np.random.geometric(0.05, 500),
        'user_type': np.random.choice(['new', 'returning'], 500)
    })
    treatment = pd.DataFrame({
        'user_id': range(500, 1000),
        'daily_sessions': np.random.poisson(2.8, 500),  # 12% lift
        'days_active': np.random.geometric(0.045, 500),
        'user_type': np.random.choice(['new', 'returning'], 500)
    })
    
    ab_analyzer = ABTestAnalyzer(control, treatment)
    results = ab_analyzer.calculate_engagement_lift('daily_sessions')
    
    print(f"  Control mean: {results['control_mean']:.2f} sessions/day")
    print(f"  Treatment mean: {results['treatment_mean']:.2f} sessions/day")
    print(f"  Lift: {results['lift_percent']:.1f}%")
    print(f"  P-value: {results['p_value']:.4f}")
    print(f"  Significant at 95%: {'✅ Yes' if results['significant_at_95'] else '❌ No'}")
    
    print("\n" + "=" * 60)
    print("Analysis complete! 🎉")
    print("=" * 60)
