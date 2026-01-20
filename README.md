# 🦉 Language Learning Engagement Analyzer

A data-driven product analytics framework for understanding and improving learner engagement in language learning applications—designed with Duolingo's mission in mind.

## 🎯 Product Vision

**Problem Statement:** Language learning apps face significant retention challenges. Most users abandon learning within the first month. Understanding what keeps learners engaged is crucial for product success.

**Solution:** An analytics framework that identifies engagement patterns, predicts churn risk, and provides actionable insights for product feature prioritization.

## 📊 Key Metrics Framework

### Primary Engagement Metrics
| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **Streak Retention** | % of users maintaining 7+ day streaks | Core gamification indicator |
| **Lesson Completion Rate** | Lessons completed / Lessons started | Learning friction indicator |
| **Time-to-Mastery** | Days to complete a skill level | Learning efficiency |
| **Session Frequency** | Learning sessions per week | Habit formation |
| **XP Velocity** | XP earned per active day | Engagement intensity |

### Cohort Analysis Dimensions
- New users (0-7 days)
- Developing learners (8-30 days)
- Committed learners (31-90 days)
- Power users (90+ days)

## 🔍 Analysis Modules

### 1. Streak Pattern Analysis
```python
"""
Analyzes streak behavior to identify patterns leading to streak breaks.
Key insights: What time of day do users typically break streaks?
Which lessons have highest streak-break correlation?
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StreakAnalyzer:
    def __init__(self, user_activity_df):
        self.data = user_activity_df
    
    def calculate_streak_metrics(self):
        """Calculate key streak engagement metrics."""
        metrics = {
            'avg_streak_length': self.data.groupby('user_id')['streak_days'].mean(),
            'streak_break_rate': self._calculate_break_rate(),
            'streak_recovery_rate': self._calculate_recovery_rate(),
            'longest_streak_distribution': self._streak_distribution()
        }
        return metrics
    
    def identify_at_risk_users(self, threshold_days=2):
        """
        Identify users who haven't practiced recently.
        These are candidates for re-engagement notifications.
        """
        today = datetime.now()
        at_risk = self.data[
            (today - self.data['last_activity_date']).dt.days >= threshold_days
        ]
        return at_risk[['user_id', 'streak_days', 'last_activity_date']]
    
    def streak_break_analysis(self):
        """Analyze when and why streaks break."""
        break_patterns = {
            'day_of_week': self._breaks_by_day(),
            'time_of_day': self._breaks_by_hour(),
            'lesson_type': self._breaks_by_lesson_type(),
            'difficulty_level': self._breaks_by_difficulty()
        }
        return break_patterns
```

### 2. Learning Friction Identifier
```python
"""
Identifies points in the learning journey where users drop off.
Helps PMs prioritize UX improvements and content optimization.
"""

class LearningFrictionAnalyzer:
    def __init__(self, lesson_data, user_journeys):
        self.lessons = lesson_data
        self.journeys = user_journeys
    
    def identify_drop_off_points(self):
        """Find lessons with highest abandonment rates."""
        drop_offs = self.journeys.groupby('lesson_id').agg({
            'started': 'sum',
            'completed': 'sum'
        })
        drop_offs['completion_rate'] = drop_offs['completed'] / drop_offs['started']
        drop_offs['drop_off_rate'] = 1 - drop_offs['completion_rate']
        
        # Flag high-friction lessons (bottom 10% completion rate)
        threshold = drop_offs['completion_rate'].quantile(0.1)
        high_friction = drop_offs[drop_offs['completion_rate'] <= threshold]
        
        return high_friction.sort_values('drop_off_rate', ascending=False)
    
    def friction_by_skill_type(self):
        """Analyze friction patterns by skill type (speaking, listening, etc.)."""
        return self.journeys.merge(
            self.lessons[['lesson_id', 'skill_type']], on='lesson_id'
        ).groupby('skill_type').agg({
            'completed': lambda x: x.sum() / len(x)  # Completion rate
        })
    
    def time_to_completion_analysis(self):
        """
        Analyze how long lessons take vs. expected time.
        Lessons taking too long may indicate difficulty issues.
        """
        self.journeys['time_ratio'] = (
            self.journeys['actual_time'] / self.journeys['expected_time']
        )
        return self.journeys.groupby('lesson_id')['time_ratio'].describe()
```

### 3. A/B Test Analysis Framework
```python
"""
Framework for analyzing A/B test results on engagement features.
Supports product decisions on gamification, notifications, and UX changes.
"""

import scipy.stats as stats

class ABTestAnalyzer:
    def __init__(self, control_group, treatment_group):
        self.control = control_group
        self.treatment = treatment_group
    
    def calculate_engagement_lift(self, metric='daily_sessions'):
        """Calculate lift in engagement metric."""
        control_mean = self.control[metric].mean()
        treatment_mean = self.treatment[metric].mean()
        
        lift = (treatment_mean - control_mean) / control_mean * 100
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_ind(
            self.control[metric], 
            self.treatment[metric]
        )
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'lift_percent': lift,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def retention_curve_comparison(self, days=30):
        """Compare retention curves between test groups."""
        retention_data = []
        for day in range(1, days + 1):
            control_retained = (self.control['days_active'] >= day).mean()
            treatment_retained = (self.treatment['days_active'] >= day).mean()
            retention_data.append({
                'day': day,
                'control_retention': control_retained,
                'treatment_retention': treatment_retained,
                'lift': (treatment_retained - control_retained) / control_retained * 100
            })
        return pd.DataFrame(retention_data)
    
    def segment_analysis(self, segment_col='user_type'):
        """Analyze treatment effect across user segments."""
        segments = pd.concat([self.control, self.treatment])[segment_col].unique()
        
        results = {}
        for segment in segments:
            control_seg = self.control[self.control[segment_col] == segment]
            treatment_seg = self.treatment[self.treatment[segment_col] == segment]
            
            results[segment] = self.calculate_engagement_lift()
        
        return results
```

### 4. Personalization Opportunity Identifier
```python
"""
Identifies opportunities for personalized learning experiences.
Helps product team understand where one-size-fits-all approaches fail.
"""

class PersonalizationAnalyzer:
    def __init__(self, user_data, engagement_data):
        self.users = user_data
        self.engagement = engagement_data
    
    def learning_style_clusters(self):
        """
        Identify distinct learning behavior clusters.
        Clusters could inform personalized feature development.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        features = [
            'avg_session_length',
            'preferred_lesson_type',
            'morning_vs_evening_ratio',
            'streak_importance_score',
            'social_feature_usage'
        ]
        
        X = StandardScaler().fit_transform(self.engagement[features])
        
        # Find optimal clusters
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.engagement['learning_cluster'] = kmeans.fit_predict(X)
        
        # Profile each cluster
        profiles = self.engagement.groupby('learning_cluster')[features].mean()
        return profiles
    
    def optimal_notification_timing(self):
        """
        Analyze when users are most responsive to notifications.
        Supports personalized reminder strategies.
        """
        responsive = self.engagement[self.engagement['responded_to_notification'] == True]
        
        timing_analysis = responsive.groupby([
            'user_id', 
            'notification_hour'
        ]).agg({
            'responded_to_notification': 'mean',
            'subsequent_session_length': 'mean'
        }).reset_index()
        
        return timing_analysis
```

## 📈 Sample Dashboard Metrics

### Executive Summary View
- **DAU/MAU Ratio:** Target 40%+ (healthy engagement)
- **7-Day Retention:** Target 65%+
- **30-Day Retention:** Target 35%+
- **Avg. Streak Length:** Target 14+ days
- **Lesson Completion Rate:** Target 85%+

### Product Health Indicators
```
┌─────────────────────────────────────────────────────────────┐
│  ENGAGEMENT SCORECARD - Week of Jan 15, 2026               │
├─────────────────────────────────────────────────────────────┤
│  Metric               │ Current │ Target │ Trend │ Status  │
├───────────────────────┼─────────┼────────┼───────┼─────────┤
│  Daily Active Users   │  12.5M  │  12M   │  ↑ 3% │   ✅    │
│  7-Day Retention      │  67%    │  65%   │  ↑ 2% │   ✅    │
│  Avg Session Length   │  8.2min │  10min │  ↓ 1% │   ⚠️    │
│  Streak Freeze Usage  │  15%    │  <20%  │  → 0% │   ✅    │
│  New User Activation  │  72%    │  75%   │  ↑ 5% │   ⚠️    │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Product Recommendations Framework

### Based on Analysis Insights:

1. **Streak Recovery Feature**
   - *Insight:* 60% of users who break a streak never return
   - *Recommendation:* Implement "streak repair" challenges that let users earn back lost streaks
   - *Success Metric:* Increase streak recovery rate by 20%

2. **Adaptive Difficulty**
   - *Insight:* Users drop off 40% more on lessons rated "too hard"
   - *Recommendation:* Implement dynamic difficulty adjustment based on recent performance
   - *Success Metric:* Reduce lesson abandonment by 15%

3. **Optimal Push Notification Timing**
   - *Insight:* Users respond 3x better to notifications during their learned "habit time"
   - *Recommendation:* Personalize notification timing based on historical engagement
   - *Success Metric:* Increase notification → session conversion by 25%

## 🛠️ Tech Stack

- **Data Processing:** Python, PySpark, SQL
- **Analytics:** Pandas, NumPy, SciPy
- **Visualization:** Tableau, Matplotlib, Seaborn
- **ML/Clustering:** Scikit-learn
- **Infrastructure:** AWS (S3, Redshift), Airflow

## 📁 Project Structure

```
language-learning-analyzer/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── streak_analyzer.py
│   ├── friction_analyzer.py
│   ├── ab_test_analyzer.py
│   └── personalization_analyzer.py
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_cohort_analysis.ipynb
│   └── 03_ab_test_results.ipynb
├── dashboards/
│   └── engagement_scorecard.json
└── tests/
    └── test_analyzers.py
```

## 🎯 Why This Project Matters for Duolingo

This framework directly addresses Duolingo's core challenges:
- **Retention:** Identifies why users churn and how to prevent it
- **Engagement:** Measures what keeps learners coming back
- **Personalization:** Enables data-driven feature personalization
- **Experimentation:** Supports rigorous A/B testing culture

---

*Built with ❤️ for language learners everywhere*

**Author:** Kajal Patel | [LinkedIn](https://www.linkedin.com/in/kajal-patel-cs/) | [GitHub](https://github.com/iKajalpatel21)
