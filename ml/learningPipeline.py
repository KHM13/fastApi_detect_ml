import traceback
import joblib
from sklearn.pipeline import Pipeline
from ml.customScaler import CustomScaler
class LearningPipeline():

    def __init__(self):
        self.pipeline: Pipeline = None
        self.estimator = []

    # 파이프라인 셋팅
    def set_pipeline(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def set_pipeline_for_file(self, pipeline_file_name):
        self.pipeline = joblib.load(pipeline_file_name)

    # 평가
    def validation(self, data):
        try:
            if self.pipeline is not None:
                result = self.pipeline.predict(data)
                return result
            else:
                print("[Error] Pipeline is None")
                return False
        except:
            print(f"[Predict Error] : {traceback.format_exc()}")

    # 평가율
    def validation_proba(self, data):
        try:
            if self.pipeline is not None:
                result = self.pipeline.predict_proba(data)
                return result
            else:
                print("[Error] Pipeline is None")
                return False
        except:
            print(f"[Predict Error] : {traceback.format_exc()}")

    # 탐지
    def predict(self, data):
        try:
            if self.pipeline is not None:
                result = self.pipeline.predict(data)[0]
                return result
            else:
                print("[Error] Pipeline is None")
                return False
        except:
            print(f"[Predict Error] : {traceback.format_exc()}")

    # 탐지율
    def predict_proba(self, data):
        try:
            if self.pipeline is not None:
                result = self.pipeline.predict_proba(data).reshape(-1, 1)[1][0]
                return result
            else:
                print("[Error] Pipeline is None")
                return False
        except:
            print(f"[Predict Error] : {traceback.format_exc()}")

