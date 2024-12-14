import sys 
import os
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig
        
    def get_data_transformer_object(self):
        '''
        function to get the data transformation done
        '''
        try:
            num_cols = ["writing_score", 'reading_score']
            cat_cols = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
                ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"numerical columns encoding and scaling completed: {num_cols}") 
            logging.info(f"categorical columns encoding completed: {cat_cols}") 
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_cols),
                    ("cat_pipelines", cat_pipeline, cat_cols)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data done")
            
            logging.info("obtaining preprocessing object")
            
            preprocessor_obj = self.get_data_transformer_object()
            
            target_column_name = "math_score"
            
            num_cols = ["writing_score", 'reading_score']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"applying preprocessing object on training dataframe and testing dataframe")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"saved preprocessing object")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_obj
                
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)