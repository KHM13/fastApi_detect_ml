from typing import Union

from fastapi import FastAPI, Request
from pydantic import BaseModel
from config.common import BASE_DIR
from elasticsearch import Elasticsearch

import json
import numpy as np
from datetime import datetime

from ml.learningPipeline import LearningPipeline

# fastApi
app = FastAPI()

# ml Model
algorithm = "RandomForestClassifier"
pkl_file_name = f"{BASE_DIR}/dask/output/npas_{algorithm}Pipeline.pkl"
lp = LearningPipeline()
lp.set_pipeline_for_file(pkl_file_name)
threshold = 0.7

# Elastic
today = datetime.today().strftime("%Y-%m-%d")
es = Elasticsearch(hosts='http://192.168.0.46:9211')

class Item(BaseModel):
    ip:str
    host:str
    referer:str
    url:str
    method:str
    country:str
    global_id:str
    userAgent:str
    cookie:str
    uri:str
    xForwaredForIP:str
    accept:str
    acceptCharset:str
    acceptEncoding:str
    acceptLanguage:str
    client_userAgentName:str
    client_osName:str
    client_deviceName:str
    #key:str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/detect1/{item_id}")
def test(item_id: str, q: Union[str, None] = None):
    temp_map = json.loads(item_id)
    temp_list = [[temp_map[x] for x in temp_map.keys()]]
    print(temp_list)
    temp_data = np.array(temp_list)
    y_pred = lp.predict(temp_data)
    return {"result" : y_pred}

@app.post("/pingTest/")
def test1(item: Item):
    return item

@app.get("/changeModel/{item_id}")
def changeModel(item_id: str, q: Union[str, None] = None):
    pkl_file_name2 = f"{BASE_DIR}/dask/output/{item_id}.pkl"

    try:
        lp.set_pipeline_for_file(pkl_file_name2)
        print(item_id + "로 변경 완료")
        return {"result" : 1}
    except:
        return {"result" : 0}


@app.post("/detect2/")
def test2(item: Item):

    temp_map = {"ip":item.ip,
                "host":item.host,
                "referer":item.referer,
                "url":item.url,
                "method":item.method,
                "country":item.country,
                "global_id":item.global_id,
                "userAgent":item.userAgent,
                "cookie":item.cookie,
                "uri":item.uri,
                "xForwaredForIP":item.xForwaredForIP,
                "accept":item.accept,
                "acceptCharset":item.acceptCharset,
                "acceptEncoding":item.acceptEncoding,
                "acceptLanguage":item.acceptLanguage,
                "client_userAgentName":item.client_userAgentName,
                "client_osName":item.client_osName,
                "client_deviceName":item.client_deviceName,
                #"key":Item.key,
                }

    y_pred = 0
    #print(temp_map)
    temp_list = [[temp_map[x] for x in temp_map.keys()]]
    temp_data = np.array(temp_list)
    #print(temp_data.astype(str))
    # y_pred = str(lp.predict(temp_data))
    y_prob = lp.predict_proba(temp_data)
    y_pred = 1 if y_prob >= threshold else 0
    print(f"predict : {str(y_pred)}, prob : {str(y_prob)}")
    return {"predict" : str(y_pred), "prob" : str(y_prob)}


@app.post("/detect3/")
async def handler(request: Request):
    temp_map = await request.json()
    gid = temp_map['gid']
    temp_map.pop('gid')
    y_pred = 0
    temp_list = [[temp_map[x] for x in temp_map.keys()]]
    temp_data = np.array(temp_list)
    y_prob = lp.predict_proba(temp_data)
    y_pred = 1 if y_prob >= threshold else 0
    
    # ES 에 넣기
    was = "fastApi_server_1"
    model = pkl_file_name + "(" + today + ")"
    data = {"WAS" : was, "model" : model, "GID" : gid, "Type" : "cp"}
    #data = {"WAS" : was, "model" : model, "GID" : gid, "Type" : "cl"}
    es.index(index="npas_ml", body=data)

    print(f"predict : {str(y_pred)}, prob : {str(y_prob)}")
    return {"predict" : str(y_pred), "prob" : str(y_prob)}




# heart

"""
age: int
sex: int
cp: int
trtbps: int
chol: int
fbs: int
restecg: int
thalachh: int
exng: int
oldpeak: float
slp: int
caa: int
thall: int
key: str

temp_map = {"age" : item.age,
                "sex" : item.sex,
                "cp" : item.cp,
                "trtbps" : item.trtbps,
                "chol" : item.chol,
                "fbs" : item.fbs,
                "restecg" : item.restecg,
                "thalachh" : item.thalachh,
                "exng" : item.exng,
                "oldpeak" : item.oldpeak,
                "slp" : item.slp,
                "caa" : item.caa,
                "thall" : item.thall}
"""
