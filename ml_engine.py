import pandas as pd
import numpy as np
import os
from datetime import datetime
# Heavy imports moved inside classes to speed up initial load

class WaitTimeModel:
    def __init__(self, data_path=None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = data_path if data_path else os.path.join(self.base_dir, 'Quiz_25-26.csv')
        self.model_path = os.path.join(self.base_dir, 'best_model.joblib')
        self.metrics_path = os.path.join(self.base_dir, 'model_metrics.joblib')
        self.pipeline = None
        self.best_model_name = None
        self.metrics = None
        self.df = None

    def load_data(self):
        if not os.path.exists(self.data_path):
            return None
        self.df = pd.read_csv(self.data_path)
        # Basic cleaning
        self.df.dropna(subset=['Wait_Time_Minutes'], inplace=True)
        return self.df

    def train(self):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        import joblib
        
        df = self.load_data()
        if df is None:
            return False

        X = df.drop(columns=['Wait_Time_Minutes'])
        y = df['Wait_Time_Minutes']

        # Define features
        categorical_features = ['Day_of_Week']
        numeric_features = [
            'Hour_of_Day', 'Number_of_Customers', 'Number_of_Reservations', 
            'Staff_On_Duty', 'Is_Holiday', 'Weather_Score', 'Average_Meal_Prep_Time_Minutes'
        ]

        # Preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42)
        }

        best_score = -np.inf
        results = {}

        for name, model in models.items():
            pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            
            score = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results[name] = {
                'R2': score,
                'MAE': mae,
                'RMSE': rmse,
                'model': pipe,
                'actual': y_test.values,
                'pred': y_pred
            }

            if score > best_score:
                best_score = score
                self.pipeline = pipe
                self.best_model_name = name
                self.metrics = (score, mae, rmse)

        # Save best model
        joblib.dump(self.pipeline, self.model_path)
        
        # Save all metrics for comparison (excluding the model object itself for joblib convenience)
        comparison_table = pd.DataFrame({k: {m: v[m] for m in ['R2', 'MAE', 'RMSE']} for k, v in results.items()}).T
        
        # Save diagnostic data from best model
        best_results = results[self.best_model_name]
        
        joblib.dump({
            'best_name': self.best_model_name,
            'metrics': self.metrics,
            'comparison': comparison_table,
            'actual': best_results['actual'],
            'pred': best_results['pred']
        }, self.metrics_path)

        return True

    def predict(self, input_data):
        import joblib
        if self.pipeline is None:
            if os.path.exists(self.model_path):
                self.pipeline = joblib.load(self.model_path)
            else:
                return None
        
        prediction = self.pipeline.predict(input_data)
        return prediction[0]

    def get_risk_analysis(self, prediction, customers, staff):
        risk = "Low"
        recommendations = []
        factors = []

        if prediction > 45:
            risk = "High"
            recommendations.append("Alert: Immediate staff increase recommended to mitigate extreme delays.")
        elif prediction > 25:
            risk = "Medium"
            recommendations.append("Consider opening an additional service station for the next 60 minutes.")
        else:
            recommendations.append("Operations normal. Focus on maintains current service speed.")

        if customers > 50 and staff < 6:
            factors.append("Critically low staff-to-customer ratio.")
            recommendations.append("Strategic staff call-in required for peak volume management.")
        
        if not recommendations:
            recommendations.append("Maintain standard operating procedures.")

        return {
            "risk": risk,
            "recommendations": recommendations,
            "factors": factors
        }

    def get_shift_performance(self):
        df = self.load_data()
        if df is None:
            return None
        
        # Define shifts: Morning < 15 (3 PM), Evening >= 15
        df['Shift'] = df['Hour_of_Day'].apply(lambda x: 'Morning' if x < 15 else 'Evening')
        shift_avg = df.groupby('Shift')['Wait_Time_Minutes'].mean().to_dict()
        
        # Calculate morning/evening data for charts
        morning_data = df[df['Shift'] == 'Morning'].sort_values('Hour_of_Day')
        evening_data = df[df['Shift'] == 'Evening'].sort_values('Hour_of_Day')
        
        return {
            'averages': shift_avg,
            'morning_series': morning_data,
            'evening_series': evening_data
        }

    def get_forecast_preview(self):
        # Generate a sample prediction for the next hour based on dataset averages
        df = self.load_data()
        if df is None:
            return 20.0
        
        current_hour = datetime.now().hour
        next_hour = (current_hour + 1) % 24
        
        # Create a "typical" scenario for the next hour
        recent_bench = df[df['Hour_of_Day'] == next_hour]
        if recent_bench.empty:
            recent_bench = df
            
        input_data = pd.DataFrame({
            'Day_of_Week': [datetime.now().strftime('%A')],
            'Hour_of_Day': [next_hour],
            'Number_of_Customers': [int(recent_bench['Number_of_Customers'].mean())],
            'Number_of_Reservations': [int(recent_bench['Number_of_Reservations'].mean())],
            'Staff_On_Duty': [int(recent_bench['Staff_On_Duty'].mean())],
            'Is_Holiday': [0], # Simplified
            'Weather_Score': [2], # "Clear"
            'Average_Meal_Prep_Time_Minutes': [int(recent_bench['Average_Meal_Prep_Time_Minutes'].mean())]
        })
        
        return self.predict(input_data)
        if self.pipeline is None:
            if os.path.exists(self.model_path):
                self.pipeline = joblib.load(self.model_path)
            else:
                return None
        
        regressor = self.pipeline.named_steps['regressor']
        if hasattr(regressor, 'feature_importances_'):
            importances = regressor.feature_importances_
            
            # Get feature names after transformation
            preprocessor = self.pipeline.named_steps['preprocessor']
            cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['Day_of_Week'])
            feature_names = [
                'Hour_of_Day', 'Number_of_Customers', 'Number_of_Reservations', 
                'Staff_On_Duty', 'Is_Holiday', 'Weather_Score', 'Average_Meal_Prep_Time_Minutes'
            ] + list(cat_features)
            
            # Pad or trim to match importances length (sometimes OneHot produces unexpected counts if unique values vary)
            if len(feature_names) != len(importances):
                feature_names = feature_names[:len(importances)]
            
            return pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        return None

if __name__ == "__main__":
    engine = WaitTimeModel()
    engine.train()
    print(f"Trained model: {engine.best_model_name}")
    print(f"R2 Score: {engine.metrics[0]:.4f}")
